//! Per-frame draw loop. `draw_frame` acquires a swapchain image, batches
//! instances per material (grouped by texture for sampler materials, by mesh
//! otherwise), uploads the per-instance bytes, records the command buffer via
//! `record_material_command_buffer`, submits, and presents.

use ash::vk;
use glam::Mat4;

use std::collections::HashMap;
use std::ptr;

use super::constants::MAX_FRAMES_IN_FLIGHT;
use super::{
    GraphicsManager, MaterialHandle, MaterialInstanceData, MeshHandle, ModelHandle, TextureHandle,
};

/// Internal per-material draw record produced inside `draw_frame`.
struct MaterialBatch {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    instance_buffer: vk::Buffer,
    draws: Vec<MaterialDraw>,
}

struct MaterialDraw {
    descriptor_set: vk::DescriptorSet,
    mesh_vertex_buffer: vk::Buffer,
    mesh_index_buffer: vk::Buffer,
    mesh_index_count: u32,
    instance_offset: u64,
    instance_count: u32,
}

fn append_instance_bytes(out: &mut Vec<u8>, instance_stride: u32, inst: &MaterialInstanceData) {
    let stride = instance_stride as usize;
    let start = out.len();
    out.resize(start + stride, 0);
    let dst = &mut out[start..start + stride];
    // Model matrix: 64 bytes at offset 0. glam's Mat4 is #[repr(C)] of 16 f32
    // columns; a raw byte copy is layout-equivalent to the previous typed copy.
    let m_bytes: &[u8] = unsafe {
        ::std::slice::from_raw_parts(
            (&inst.last_model as *const Mat4) as *const u8,
            ::std::mem::size_of::<Mat4>(),
        )
    };
    dst[0..64].copy_from_slice(m_bytes);
    if !inst.extra.is_empty() {
        dst[64..64 + inst.extra.len()].copy_from_slice(&inst.extra);
    }
}

fn record_material_command_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    framebuffer: vk::Framebuffer,
    render_pass: vk::RenderPass,
    surface_extent: vk::Extent2D,
    batches: &[MaterialBatch],
) {
    let begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        p_inheritance_info: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    // Attachment order must match `vulkan::create_render_pass`: colour at 0,
    // depth at 1.
    let clear_values = [
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];

    let render_pass_begin_info = vk::RenderPassBeginInfo {
        s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
        p_next: ptr::null(),
        render_pass,
        framebuffer,
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: surface_extent,
        },
        clear_value_count: clear_values.len() as u32,
        p_clear_values: clear_values.as_ptr(),
        ..Default::default()
    };

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("Failed to begin recording Command Buffer!");
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );

        for batch in batches.iter() {
            if batch.draws.is_empty() {
                continue;
            }
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                batch.pipeline,
            );
            let mut current_set = vk::DescriptorSet::null();
            for d in batch.draws.iter() {
                if d.descriptor_set != current_set {
                    let sets = [d.descriptor_set];
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        batch.pipeline_layout,
                        0,
                        &sets,
                        &[],
                    );
                    current_set = d.descriptor_set;
                }
                let vertex_buffers = [d.mesh_vertex_buffer, batch.instance_buffer];
                let offsets = [0_u64, d.instance_offset];
                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    d.mesh_index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(
                    command_buffer,
                    d.mesh_index_count,
                    d.instance_count,
                    0,
                    0,
                    0,
                );
            }
        }

        device.cmd_end_render_pass(command_buffer);

        device
            .end_command_buffer(command_buffer)
            .expect("Failed to record Command Buffer at Ending!");
    }
}

impl GraphicsManager {
    pub fn draw_frame(&mut self, transforms: &[(ModelHandle, Mat4)]) {
        #[cfg(feature = "hot-reload")]
        self.process_hot_reloads();

        let wait_fences = [self.in_flight_fences[self.current_frame]];

        unsafe {
            self.device
                .wait_for_fences(&wait_fences, true, u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            match result {
                Ok(image_index) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain();
                        return;
                    }
                    _ => panic!("Failed to acquire Swap Chain Image!"),
                },
            }
        };

        // A handle in `transforms` that isn't registered is a programming error → panic.
        // A registered handle missing from `transforms` keeps last frame's matrix.
        for (handle, transform) in transforms {
            match self.instances.get_mut(handle) {
                Some(inst) => inst.last_model = *transform,
                None => panic!("draw_frame received transform for unknown ModelHandle"),
            }
        }

        // Per-material batching. Iterate materials in registration order
        // (MaterialHandle order) — this preserves today's "solid first,
        // textured second" paint order because the engine registers them in
        // that order during `new`.
        let mut batches: Vec<MaterialBatch> = Vec::new();
        let mut material_handles: Vec<MaterialHandle> = self.materials.keys().copied().collect();
        material_handles.sort();

        for mh in material_handles {
            let mat = &self.materials[&mh];
            // Gather instances owned by this material.
            let mut owned: Vec<(ModelHandle, &MaterialInstanceData)> = self
                .instances
                .iter()
                .filter(|(_, ii)| ii.material == mh)
                .map(|(h, ii)| (*h, ii))
                .collect();
            if owned.is_empty() {
                continue;
            }
            owned.sort_by_key(|(h, _)| *h);

            // Pack instance bytes (model + extra) contiguously, ordered by
            // group (mesh for non-sampler, texture for sampler).
            let mut packed: Vec<u8> = Vec::with_capacity(owned.len() * mat.instance_stride as usize);
            let mut draws: Vec<MaterialDraw> = Vec::new();

            if mat.has_sampler {
                // Group by texture.
                let mut by_texture: HashMap<TextureHandle, Vec<(ModelHandle, &MaterialInstanceData)>> =
                    HashMap::new();
                for (h, ii) in owned.iter() {
                    if let Some(t) = ii.texture {
                        by_texture.entry(t).or_default().push((*h, ii));
                    }
                }
                let mut tex_handles: Vec<TextureHandle> = by_texture.keys().copied().collect();
                tex_handles.sort_by_key(|t| t.0);
                for th in tex_handles {
                    let group = by_texture.get_mut(&th).unwrap();
                    group.sort_by_key(|(h, _)| *h);
                    let offset_bytes = packed.len() as u64;
                    let count = group.len() as u32;
                    for (_, ii) in group.iter() {
                        append_instance_bytes(&mut packed, mat.instance_stride, ii);
                    }
                    let sets = mat
                        .texture_descriptor_sets
                        .get(&th)
                        .expect("draw_frame: missing per-texture descriptor sets");
                    draws.push(MaterialDraw {
                        descriptor_set: sets[image_index as usize],
                        mesh_vertex_buffer: self.textured_quad_vertex_buffer,
                        mesh_index_buffer: self.textured_quad_index_buffer,
                        mesh_index_count: self.textured_quad_index_count,
                        instance_offset: offset_bytes,
                        instance_count: count,
                    });
                }
            } else {
                // Group by mesh.
                let mut by_mesh: HashMap<MeshHandle, Vec<(ModelHandle, &MaterialInstanceData)>> =
                    HashMap::new();
                for (h, ii) in owned.iter() {
                    if let Some(m) = ii.mesh {
                        by_mesh.entry(m).or_default().push((*h, ii));
                    }
                }
                let mut mesh_handles: Vec<MeshHandle> = by_mesh.keys().copied().collect();
                mesh_handles.sort();
                let camera_set = mat.camera_descriptor_sets[image_index as usize];
                for mhk in mesh_handles {
                    let group = by_mesh.get_mut(&mhk).unwrap();
                    group.sort_by_key(|(h, _)| *h);
                    let mesh = &self.meshes[&mhk];
                    let offset_bytes = packed.len() as u64;
                    let count = group.len() as u32;
                    for (_, ii) in group.iter() {
                        append_instance_bytes(&mut packed, mat.instance_stride, ii);
                    }
                    draws.push(MaterialDraw {
                        descriptor_set: camera_set,
                        mesh_vertex_buffer: mesh.vertex_buffer,
                        mesh_index_buffer: mesh.index_buffer,
                        mesh_index_count: mesh.index_count,
                        instance_offset: offset_bytes,
                        instance_count: count,
                    });
                }
            }

            if !packed.is_empty() {
                let buffer_size = packed.len() as vk::DeviceSize;
                unsafe {
                    let data_ptr = self
                        .device
                        .map_memory(
                            mat.instance_buffer_memories[self.current_frame],
                            0,
                            buffer_size,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to map material instance buffer memory")
                        as *mut u8;
                    data_ptr.copy_from_nonoverlapping(packed.as_ptr(), packed.len());
                    self.device
                        .unmap_memory(mat.instance_buffer_memories[self.current_frame]);
                }
            }

            batches.push(MaterialBatch {
                pipeline: mat.pipeline,
                pipeline_layout: mat.pipeline_layout,
                instance_buffer: mat.instance_buffers[self.current_frame],
                draws,
            });
        }

        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");
        }
        // Draw order is load-bearing: the render pass has no depth attachment
        // and the pipelines don't blend, so command order *is* paint order.
        // Materials are iterated in registration order so that game code can
        // control overdraw layering by the order it registers materials in.
        record_material_command_buffer(
            &self.device,
            command_buffer,
            self.swapchain_framebuffers[image_index as usize],
            self.render_pass,
            self.swapchain_extent,
            &batches,
        );

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        }];

        unsafe {
            self.device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.swapchain];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };

        let is_resized = match result {
            Ok(_) => self.is_framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.is_framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}
