//! Mesh / texture / material / per-instance registration and teardown. All the
//! `register_*` and `unregister_*` methods live here. Engine-internal callers
//! (the `resources` layer) reach mesh/texture registration through `pub(crate)`
//! entry points; material and per-instance registration is public.

use ash::vk;
use glam::Mat4;

use std::collections::HashMap;

use super::constants::MAX_FRAMES_IN_FLIGHT;
use super::material::{
    allocate_camera_descriptor_sets, allocate_textured_descriptor_sets,
    create_descriptor_set_layout as create_material_descriptor_set_layout,
    create_pipeline as create_material_pipeline, Material, MAX_INSTANCES_PER_MATERIAL,
};
use super::structures::ModelMesh;
use super::{
    vulkan, GraphicsManager, MaterialHandle, MaterialInstanceData, MeshBuffers, MeshHandle,
    ModelHandle, TextureHandle, TextureResources, VertexAttr, MAX_MATERIALS,
};
use super::MaterialDesc;

impl GraphicsManager {
    /// Register a new material. The renderer copies the SPIR-V bytes and the
    /// attribute / binding lists; the caller's references are no longer needed
    /// after the call returns.
    ///
    /// The first instance attribute **must** be [`VertexAttr::Mat4`] — the
    /// engine writes the per-instance model matrix at that offset. Anything
    /// after it is material-defined and is supplied as raw bytes through
    /// [`Self::register_material_instance`].
    pub fn register_material(&mut self, desc: &MaterialDesc) -> MaterialHandle {
        assert!(
            !desc.instance_attrs.is_empty()
                && matches!(desc.instance_attrs[0], VertexAttr::Mat4),
            "register_material: first instance attribute must be VertexAttr::Mat4 (the model matrix)"
        );
        assert!(
            self.materials.len() < MAX_MATERIALS,
            "register_material: hit MAX_MATERIALS cap (bump in graphics_manager::MAX_MATERIALS)"
        );

        let handle = MaterialHandle(self.next_material_handle);
        self.next_material_handle += 1;

        // Each material samples at most one camera UBO; the slot it declares
        // picks which one. Allocate (or reuse) per-swapchain-image UBOs for
        // that slot before we wire descriptor sets against them.
        let camera_slot = Material::camera_slot(desc.bindings).expect(
            "register_material: every material must declare a Binding::CameraUbo(slot)",
        );
        self.ensure_camera_slot(camera_slot);

        let descriptor_set_layout =
            create_material_descriptor_set_layout(&self.device, desc.bindings);
        let (pipeline, pipeline_layout, vertex_stride, instance_stride) = create_material_pipeline(
            &self.device,
            self.render_pass,
            self.swapchain_extent,
            descriptor_set_layout,
            self.pipeline_cache,
            desc.vertex_spv,
            desc.fragment_spv,
            desc.vertex_attrs,
            desc.instance_attrs,
            desc.depth,
            desc.blend,
        );
        // Per-instance buffer: host-visible+coherent, rewritten each frame.
        let buffer_size =
            (MAX_INSTANCES_PER_MATERIAL * instance_stride as usize) as vk::DeviceSize;
        let mut instance_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut instance_buffer_memories = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer, memory) = vulkan::create_buffer(
                &self.device,
                buffer_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                &self.physical_device_memory_properties,
            );
            instance_buffers.push(buffer);
            instance_buffer_memories.push(memory);
        }

        let has_sampler = Material::has_sampler_binding(desc.bindings);
        // Pre-allocate camera-only descriptor sets for non-sampler materials
        // (the textured-batch materials get per-(material, texture) sets on
        // demand when a textured instance is first registered).
        let camera_descriptor_sets = if has_sampler {
            Vec::new()
        } else {
            allocate_camera_descriptor_sets(
                &self.device,
                self.descriptor_pool,
                descriptor_set_layout,
                &self.camera_uniform_buffers[&camera_slot],
            )
        };

        // Mat4 is always first in the instance stream (64 bytes); anything
        // after it is the material-specific "extra" payload.
        let extra_size = instance_stride - 64;

        let _ = vertex_stride; // currently unused outside pipeline creation
        self.materials.insert(
            handle,
            Material {
                vertex_spv: desc.vertex_spv.to_vec(),
                fragment_spv: desc.fragment_spv.to_vec(),
                vertex_attrs: desc.vertex_attrs.to_vec(),
                instance_attrs: desc.instance_attrs.to_vec(),
                instance_stride,
                extra_size,
                depth: desc.depth,
                blend: desc.blend,
                camera_slot,
                descriptor_set_layout,
                pipeline_layout,
                pipeline,
                instance_buffers,
                instance_buffer_memories,
                has_sampler,
                camera_descriptor_sets,
                texture_descriptor_sets: HashMap::new(),
            },
        );

        handle
    }

    pub(crate) fn register_mesh(&mut self, mesh: &ModelMesh) -> MeshHandle {
        let handle = MeshHandle(self.next_mesh_handle);
        self.next_mesh_handle += 1;
        // `mesh.vertex_bytes` is a raw byte buffer; the material that draws
        // against this mesh owns the layout via its `vertex_attrs`. Stride
        // isn't passed to the buffer creator (the GPU only sees raw bytes),
        // but the material's pipeline computes the binding stride from the
        // same `vertex_attrs` so the two agree by construction.
        let _ = mesh.vertex_stride;
        let (vertex_buffer, vertex_memory) = vulkan::create_vertex_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.vertex_bytes,
        );
        let (index_buffer, index_memory) = vulkan::create_index_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.indices,
        );
        self.meshes.insert(
            handle,
            MeshBuffers {
                vertex_buffer,
                vertex_memory,
                index_buffer,
                index_memory,
                index_count: mesh.indices.len() as u32,
            },
        );
        handle
    }

    /// Register an instance of a material. The variant is selected by
    /// `mesh` / `texture`:
    ///
    /// - Non-sampler material: pass `mesh = Some(_)`, `texture = None`.
    /// - Sampler material: pass `mesh = None`, `texture = Some(_)` (the
    ///   engine's shared unit quad is used as the geometry).
    ///
    /// `extra` must be exactly the size declared by the material's
    /// `instance_attrs` after the leading [`VertexAttr::Mat4`].
    pub fn register_material_instance(
        &mut self,
        material: MaterialHandle,
        mesh: Option<MeshHandle>,
        texture: Option<TextureHandle>,
        extra: &[u8],
    ) -> ModelHandle {
        let mat = self
            .materials
            .get_mut(&material)
            .expect("register_material_instance: unknown MaterialHandle");
        assert_eq!(
            extra.len() as u32,
            mat.extra_size,
            "register_material_instance: extra payload size mismatch"
        );
        if mat.has_sampler {
            assert!(
                texture.is_some(),
                "register_material_instance: sampler material requires a TextureHandle"
            );
        } else {
            assert!(
                mesh.is_some(),
                "register_material_instance: non-sampler material requires a MeshHandle"
            );
        }
        if let Some(m) = mesh {
            assert!(
                self.meshes.contains_key(&m),
                "register_material_instance: unknown MeshHandle"
            );
        }

        // Lazily allocate (material, texture) descriptor sets the first time
        // we see this pairing. The camera UBO bound at descriptor binding 0
        // is the one for this material's declared camera slot.
        if let Some(t) = texture {
            let tex = self
                .textures
                .get(&t)
                .expect("register_material_instance: unknown TextureHandle");
            if !mat.texture_descriptor_sets.contains_key(&t) {
                let camera_buffers = self
                    .camera_uniform_buffers
                    .get(&mat.camera_slot)
                    .expect("register_material_instance: material's camera slot was not initialised");
                let sets = allocate_textured_descriptor_sets(
                    &self.device,
                    self.descriptor_pool,
                    mat.descriptor_set_layout,
                    camera_buffers,
                    tex.view,
                    tex.sampler,
                );
                mat.texture_descriptor_sets.insert(t, sets);
            }
        }

        let handle = ModelHandle(self.next_handle);
        self.next_handle += 1;
        self.instances.insert(
            handle,
            MaterialInstanceData {
                material,
                mesh,
                texture,
                extra: extra.to_vec(),
                last_model: Mat4::IDENTITY,
            },
        );
        handle
    }

    /// Convenience wrapper for the built-in solid material. Equivalent to
    /// `register_material_instance(solid_material(), Some(mesh), None,
    /// bytemuck-cast(color))`.
    pub fn register_instance(&mut self, mesh: MeshHandle, color: [f32; 3]) -> ModelHandle {
        let bytes: [u8; 12] = unsafe { ::std::mem::transmute(color) };
        let material = self.solid_material_handle;
        self.register_material_instance(material, Some(mesh), None, &bytes)
    }

    /// Convenience wrapper for the built-in textured material.
    pub fn register_textured_instance(
        &mut self,
        texture: TextureHandle,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) -> ModelHandle {
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(unsafe {
            &::std::mem::transmute::<[f32; 2], [u8; 8]>(uv_offset)
        });
        bytes[8..16].copy_from_slice(unsafe {
            &::std::mem::transmute::<[f32; 2], [u8; 8]>(uv_scale)
        });
        let material = self.textured_material_handle;
        self.register_material_instance(material, None, Some(texture), &bytes)
    }

    /// Replace the per-instance "extra" payload (everything after the model
    /// matrix) for an already-registered instance. The next `draw_frame`
    /// re-packs the instance buffer and the new bytes are picked up by the
    /// vertex shader.
    ///
    /// Bytes must be exactly the size declared by the owning material's
    /// `instance_attrs` after the leading [`VertexAttr::Mat4`]. Useful for
    /// occasional state changes (e.g. a menu option's tint colour) without
    /// the cost of unregister + re-register.
    pub fn set_instance_extra(&mut self, handle: ModelHandle, extra: &[u8]) {
        let inst = self
            .instances
            .get_mut(&handle)
            .expect("set_instance_extra: unknown ModelHandle");
        assert_eq!(
            extra.len(),
            inst.extra.len(),
            "set_instance_extra: payload size mismatch (material declared {} bytes)",
            inst.extra.len()
        );
        inst.extra.copy_from_slice(extra);
    }

    pub(crate) fn unregister_mesh(&mut self, mesh: MeshHandle) {
        if let Some(buffers) = self.meshes.remove(&mesh) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_mesh");
            }
            super::destroy_mesh(&self.device, &buffers);
        }
    }

    pub fn unregister_instance(&mut self, handle: ModelHandle) {
        self.instances.remove(&handle);
    }

    /// Alias kept for source compatibility with pre-Phase-7 game code. Today
    /// every instance — solid or textured — lives in one `instances` map keyed
    /// by [`ModelHandle`], so the two unregister paths are identical.
    pub fn unregister_textured_instance(&mut self, handle: ModelHandle) {
        self.unregister_instance(handle);
    }

    pub(crate) fn register_texture(&mut self, png_bytes: &[u8]) -> TextureHandle {
        let handle = TextureHandle(self.next_texture_handle);
        self.next_texture_handle += 1;

        let (image, memory, _w, _h) = vulkan::load_texture_image(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            png_bytes,
        );
        let view = vulkan::create_image_view(
            &self.device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        let sampler = vulkan::create_texture_sampler(&self.device);

        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler },
        );
        handle
    }

    pub(crate) fn register_texture_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> TextureHandle {
        let handle = TextureHandle(self.next_texture_handle);
        self.next_texture_handle += 1;

        let (image, memory) = vulkan::upload_rgba_image(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            width,
            height,
            rgba,
        );
        let view = vulkan::create_image_view(
            &self.device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        let sampler = vulkan::create_texture_sampler(&self.device);

        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler },
        );
        handle
    }

    pub(crate) fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some(tex) = self.textures.remove(&handle) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_texture");
            }
            // Every material that has allocated descriptor sets against this
            // texture must release them back to the pool.
            for mat in self.materials.values_mut() {
                if let Some(sets) = mat.texture_descriptor_sets.remove(&handle) {
                    if !sets.is_empty() {
                        unsafe {
                            let _ = self
                                .device
                                .free_descriptor_sets(self.descriptor_pool, &sets);
                        }
                    }
                }
            }
            super::destroy_texture_resources(&self.device, &tex);
        }
    }
}

