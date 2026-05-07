pub mod constants;
pub mod debug;
pub mod fps_limiter;
pub mod platforms;
pub mod share;
pub mod structures;
pub mod tools;
pub mod window;

use cgmath::Deg;
use cgmath::Matrix4;
use cgmath::Point3;
use cgmath::Vector3;
use constants::*;
use structures::{QueueFamilyIndices, SurfaceStuff};

use ash::version::DeviceV1_0;
use ash::version::InstanceV1_0;
use ash::vk;

use std::collections::HashMap;
use std::ptr;

use self::structures::{ModelMesh, TexturedModelMesh, UniformBufferObject};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ModelHandle(u32);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TextureHandle(u32);

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[allow(dead_code)]
pub enum PipelineKind {
    SolidColour,
    Textured,
}

pub struct TextureResources {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
}

pub struct ModelBuffers {
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    index_count: u32,

    // Last model matrix supplied via draw_frame's transforms slice. Preserves the
    // documented behaviour: a registered handle missing from `transforms` redraws
    // with whatever it was last given (identity at registration).
    last_model: Matrix4<f32>,

    pipeline: PipelineKind,
    // Textured-only: one descriptor set per swapchain image, binding the per-image
    // camera UBO + this model's texture (image view + sampler from `TextureResources`).
    // Empty for SolidColour models — they share `camera_descriptor_sets`.
    descriptor_sets: Vec<vk::DescriptorSet>,
}

fn build_model_buffers(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    mesh: &ModelMesh,
) -> ModelBuffers {
    let (vertex_buffer, vertex_buffer_memory) =
        share::create_vertex_buffer(device, mem_props, command_pool, queue, &mesh.vertices);
    let (index_buffer, index_buffer_memory) =
        share::create_index_buffer(device, mem_props, command_pool, queue, &mesh.indices);

    ModelBuffers {
        vertex_buffer,
        vertex_buffer_memory,
        index_buffer,
        index_buffer_memory,
        index_count: mesh.indices.len() as u32,
        last_model: Matrix4::from_scale(1.0),
        pipeline: PipelineKind::SolidColour,
        descriptor_sets: Vec::new(),
    }
}

fn destroy_model_buffers(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    buffers: &ModelBuffers,
) {
    unsafe {
        if !buffers.descriptor_sets.is_empty() {
            // Pool was created with FREE_DESCRIPTOR_SET so this is legal.
            device.free_descriptor_sets(descriptor_pool, &buffers.descriptor_sets);
        }

        device.destroy_buffer(buffers.index_buffer, None);
        device.free_memory(buffers.index_buffer_memory, None);

        device.destroy_buffer(buffers.vertex_buffer, None);
        device.free_memory(buffers.vertex_buffer_memory, None);
    }
}

fn destroy_texture_resources(device: &ash::Device, tex: &TextureResources) {
    unsafe {
        device.destroy_sampler(tex.sampler, None);
        device.destroy_image_view(tex.view, None);
        device.destroy_image(tex.image, None);
        device.free_memory(tex.memory, None);
    }
}

fn write_camera_ubos(
    device: &ash::Device,
    memories: &[vk::DeviceMemory],
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
) {
    let ubo = UniformBufferObject { view, proj };
    let buffer_size = ::std::mem::size_of::<UniformBufferObject>() as u64;
    for &memory in memories {
        unsafe {
            let data_ptr = device
                .map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .expect("Failed to Map Memory") as *mut UniformBufferObject;
            data_ptr.copy_from_nonoverlapping(&ubo, 1);
            device.unmap_memory(memory);
        }
    }
}

pub struct GraphicsManager {
    window: winit::window::Window,

    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_merssager: vk::DebugUtilsMessengerEXT,

    physical_device: vk::PhysicalDevice,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,

    queue_family: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    ubo_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    // Second pipeline for textured (sampled image) draws. Coexists with the
    // solid-colour pipeline in the same render pass; per-model dispatch in
    // `record_command_buffer` picks one based on `ModelBuffers::pipeline`.
    textured_ubo_layout: vk::DescriptorSetLayout,
    textured_pipeline_layout: vk::PipelineLayout,
    textured_pipeline: vk::Pipeline,

    // Camera UBO `(view, proj)` is shared across all models — one buffer per
    // swapchain image, one descriptor set per swapchain image. Contents are
    // rewritten only when the camera changes (init, swapchain recreation).
    camera_uniform_buffers: Vec<vk::Buffer>,
    camera_uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    camera_descriptor_sets: Vec<vk::DescriptorSet>,

    model_buffers: HashMap<ModelHandle, ModelBuffers>,
    next_handle: u32,
    // Texture lifetime is independent of model lifetime — multiple textured
    // models may share one texture (e.g. a font atlas with one glyph per quad).
    textures: HashMap<TextureHandle, TextureResources>,
    next_texture_handle: u32,
    current_view: Matrix4<f32>,
    current_proj: Matrix4<f32>,

    command_pool: vk::CommandPool,
    // MAX_FRAMES_IN_FLIGHT command buffers, indexed by current_frame. Re-recorded each
    // frame in draw_frame so the per-draw push-constant model matrix reflects the
    // current Scene transforms.
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    is_framebuffer_resized: bool,
}

impl GraphicsManager {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> GraphicsManager {
        let window = window::init_window(event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

        let entry = ash::Entry::new().unwrap();
        let instance = share::create_instance(
            &entry,
            WINDOW_TITLE,
            VALIDATION.is_enable,
            &VALIDATION.required_validation_layers,
        );
        let surface_stuff = share::create_surface(&entry, &instance, &window);
        let (debug_utils_loader, debug_merssager) =
            debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
        let physical_device =
            share::pick_physical_device(&instance, &surface_stuff, &DEVICE_EXTENSIONS);
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (device, queue_family) = share::create_logical_device(
            &instance,
            physical_device,
            &VALIDATION,
            &DEVICE_EXTENSIONS,
            &surface_stuff,
        );
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family.present_family.unwrap(), 0) };
        let swapchain_stuff = share::create_swapchain(
            &instance,
            &device,
            physical_device,
            &window,
            &surface_stuff,
            &queue_family,
        );
        let swapchain_imageviews = share::create_image_views(
            &device,
            swapchain_stuff.swapchain_format,
            &swapchain_stuff.swapchain_images,
        );
        let render_pass = share::create_render_pass(&device, swapchain_stuff.swapchain_format);
        let ubo_layout = share::create_descriptor_set_layout(&device);
        let textured_ubo_layout = share::create_textured_descriptor_set_layout(&device);
        let (graphics_pipeline, pipeline_layout) = share::create_graphics_pipeline(
            &device,
            render_pass,
            swapchain_stuff.swapchain_extent,
            ubo_layout,
        );
        let (textured_pipeline, textured_pipeline_layout) = share::create_textured_graphics_pipeline(
            &device,
            render_pass,
            swapchain_stuff.swapchain_extent,
            textured_ubo_layout,
        );
        let swapchain_framebuffers = share::create_framebuffers(
            &device,
            render_pass,
            &swapchain_imageviews,
            swapchain_stuff.swapchain_extent,
        );
        let command_pool = share::create_command_pool(&device, &queue_family);

        // Camera lives in the renderer now. View/proj are hardcoded; projection is recomputed
        // on swapchain recreation. Models are registered by Scene after `new()` returns.
        let current_view = Matrix4::look_at(
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let current_proj = cgmath::perspective(
            Deg(45.0),
            swapchain_stuff.swapchain_extent.width as f32
                / swapchain_stuff.swapchain_extent.height as f32,
            0.1,
            10.0,
        );

        let swapchain_image_count = swapchain_stuff.swapchain_images.len();
        let (camera_uniform_buffers, camera_uniform_buffers_memory) = share::create_uniform_buffers(
            &device,
            &physical_device_memory_properties,
            swapchain_image_count,
        );
        let descriptor_pool =
            share::create_descriptor_pool(&device, swapchain_image_count, MAX_TEXTURED_MODELS);
        let camera_descriptor_sets = share::create_descriptor_sets(
            &device,
            descriptor_pool,
            ubo_layout,
            &camera_uniform_buffers,
            swapchain_image_count,
        );

        write_camera_ubos(
            &device,
            &camera_uniform_buffers_memory,
            current_view,
            current_proj,
        );

        let model_buffers: HashMap<ModelHandle, ModelBuffers> = HashMap::new();
        let next_handle: u32 = 0;

        let command_buffers =
            share::allocate_command_buffers(&device, command_pool, MAX_FRAMES_IN_FLIGHT as u32);
        let sync_ojbects = share::create_sync_objects(&device, MAX_FRAMES_IN_FLIGHT);

        GraphicsManager {
            window,

            _entry: entry,
            instance,
            surface: surface_stuff.surface,
            surface_loader: surface_stuff.surface_loader,
            debug_utils_loader,
            debug_merssager,

            physical_device,
            physical_device_memory_properties,
            device,

            queue_family,
            graphics_queue,
            present_queue,

            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            swapchain_format: swapchain_stuff.swapchain_format,
            swapchain_images: swapchain_stuff.swapchain_images,
            swapchain_extent: swapchain_stuff.swapchain_extent,
            swapchain_imageviews,
            swapchain_framebuffers,

            pipeline_layout,
            render_pass,
            graphics_pipeline,
            ubo_layout,
            textured_ubo_layout,
            textured_pipeline_layout,
            textured_pipeline,

            camera_uniform_buffers,
            camera_uniform_buffers_memory,
            descriptor_pool,
            camera_descriptor_sets,

            model_buffers,
            next_handle,
            textures: HashMap::new(),
            next_texture_handle: 0,
            current_view,
            current_proj,

            command_pool,
            command_buffers,

            image_available_semaphores: sync_ojbects.image_available_semaphores,
            render_finished_semaphores: sync_ojbects.render_finished_semaphores,
            in_flight_fences: sync_ojbects.inflight_fences,
            current_frame: 0,

            is_framebuffer_resized: false,
        }
    }

    pub fn window_request_redraw(&mut self) {
        self.window.request_redraw();
    }

    pub fn device_wait_idle(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
    }

    pub fn register_model(&mut self, mesh: &ModelMesh) -> ModelHandle {
        let handle = ModelHandle(self.next_handle);
        self.next_handle += 1;
        let buffers = build_model_buffers(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            mesh,
        );
        self.model_buffers.insert(handle, buffers);
        handle
    }

    #[allow(dead_code)]
    pub fn unregister_model(&mut self, handle: ModelHandle) {
        if let Some(buffers) = self.model_buffers.remove(&handle) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_model");
            }
            destroy_model_buffers(&self.device, self.descriptor_pool, &buffers);
        }
    }

    #[allow(dead_code)]
    pub fn register_texture(&mut self, png_bytes: &[u8]) -> TextureHandle {
        let handle = TextureHandle(self.next_texture_handle);
        self.next_texture_handle += 1;

        let (image, memory, _w, _h) = share::load_texture_image(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            png_bytes,
        );
        let view = share::create_image_view(
            &self.device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        let sampler = share::create_texture_sampler(&self.device);

        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler },
        );
        handle
    }

    pub fn register_texture_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> TextureHandle {
        let handle = TextureHandle(self.next_texture_handle);
        self.next_texture_handle += 1;

        let (image, memory) = share::upload_rgba_image(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            width,
            height,
            rgba,
        );
        let view = share::create_image_view(
            &self.device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        let sampler = share::create_texture_sampler(&self.device);

        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler },
        );
        handle
    }

    #[allow(dead_code)]
    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some(tex) = self.textures.remove(&handle) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_texture");
            }
            destroy_texture_resources(&self.device, &tex);
        }
    }

    #[allow(dead_code)]
    pub fn register_textured_model_with(
        &mut self,
        mesh: &TexturedModelMesh,
        texture: TextureHandle,
    ) -> ModelHandle {
        let tex = self
            .textures
            .get(&texture)
            .expect("register_textured_model_with received unknown TextureHandle");

        let (vertex_buffer, vertex_buffer_memory) = share::create_vertex_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.vertices,
        );
        let (index_buffer, index_buffer_memory) = share::create_index_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.indices,
        );

        let descriptor_sets = share::create_textured_descriptor_sets(
            &self.device,
            self.descriptor_pool,
            self.textured_ubo_layout,
            &self.camera_uniform_buffers,
            tex.view,
            tex.sampler,
        );

        let buffers = ModelBuffers {
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            index_count: mesh.indices.len() as u32,
            last_model: Matrix4::from_scale(1.0),
            pipeline: PipelineKind::Textured,
            descriptor_sets,
        };

        let handle = ModelHandle(self.next_handle);
        self.next_handle += 1;
        self.model_buffers.insert(handle, buffers);
        handle
    }

    pub fn draw_frame(&mut self, transforms: &[(ModelHandle, Matrix4<f32>)]) {
        let wait_fences = [self.in_flight_fences[self.current_frame]];

        unsafe {
            self.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
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

        // Update each touched model's last_model. Handles in transforms that aren't
        // registered are a programming error → panic. Registered handles missing from
        // transforms keep last frame's model matrix.
        for (handle, transform) in transforms {
            self.model_buffers
                .get_mut(handle)
                .expect("draw_frame received transform for unknown ModelHandle")
                .last_model = *transform;
        }

        // Sorted by ModelHandle ordinal so registration order = draw order.
        // The render pass has no depth attachment and the pipeline has no alpha
        // blending, so overdraw correctness depends on this stable order.
        let mut entries: Vec<_> = self.model_buffers.iter().collect();
        entries.sort_by_key(|(handle, _)| **handle);
        let camera_set = self.camera_descriptor_sets[image_index as usize];
        let draws: Vec<share::Draw> = entries
            .iter()
            .map(|(_, buffers)| {
                let (pipeline, pipeline_layout, descriptor_set) = match buffers.pipeline {
                    PipelineKind::SolidColour => {
                        (self.graphics_pipeline, self.pipeline_layout, camera_set)
                    }
                    PipelineKind::Textured => (
                        self.textured_pipeline,
                        self.textured_pipeline_layout,
                        buffers.descriptor_sets[image_index as usize],
                    ),
                };
                share::Draw {
                    pipeline,
                    pipeline_layout,
                    descriptor_set,
                    buffers: *buffers,
                    model_matrix: buffers.last_model,
                }
            })
            .collect();

        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");
        }
        share::record_command_buffer(
            &self.device,
            command_buffer,
            self.swapchain_framebuffers[image_index as usize],
            self.render_pass,
            self.swapchain_extent,
            &draws,
        );

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

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

    fn recreate_swapchain(&mut self) {
        // parameters -------------
        let surface_suff = SurfaceStuff {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
        };
        // ------------------------

        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.cleanup_swapchain();

        let swapchain_stuff = share::create_swapchain(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.window,
            &surface_suff,
            &self.queue_family,
        );
        self.swapchain_loader = swapchain_stuff.swapchain_loader;
        self.swapchain = swapchain_stuff.swapchain;
        self.swapchain_images = swapchain_stuff.swapchain_images;
        self.swapchain_format = swapchain_stuff.swapchain_format;
        self.swapchain_extent = swapchain_stuff.swapchain_extent;

        // Aspect ratio changed — push the new projection through the shared UBO.
        self.current_proj = cgmath::perspective(
            Deg(45.0),
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            0.1,
            10.0,
        );
        write_camera_ubos(
            &self.device,
            &self.camera_uniform_buffers_memory,
            self.current_view,
            self.current_proj,
        );

        self.swapchain_imageviews =
            share::create_image_views(&self.device, self.swapchain_format, &self.swapchain_images);
        self.render_pass = share::create_render_pass(&self.device, self.swapchain_format);
        let (graphics_pipeline, pipeline_layout) = share::create_graphics_pipeline(
            &self.device,
            self.render_pass,
            swapchain_stuff.swapchain_extent,
            self.ubo_layout,
        );
        self.graphics_pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;

        let (textured_pipeline, textured_pipeline_layout) =
            share::create_textured_graphics_pipeline(
                &self.device,
                self.render_pass,
                swapchain_stuff.swapchain_extent,
                self.textured_ubo_layout,
            );
        self.textured_pipeline = textured_pipeline;
        self.textured_pipeline_layout = textured_pipeline_layout;

        self.swapchain_framebuffers = share::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_imageviews,
            self.swapchain_extent,
        );
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.textured_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.textured_pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);
            for &image_view in self.swapchain_imageviews.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for GraphicsManager {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.cleanup_swapchain();

            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);

            for buffers in self.model_buffers.values() {
                destroy_model_buffers(&self.device, self.descriptor_pool, buffers);
            }
            for tex in self.textures.values() {
                destroy_texture_resources(&self.device, tex);
            }

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for i in 0..self.camera_uniform_buffers.len() {
                self.device
                    .destroy_buffer(self.camera_uniform_buffers[i], None);
                self.device
                    .free_memory(self.camera_uniform_buffers_memory[i], None);
            }

            self.device
                .destroy_descriptor_set_layout(self.ubo_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.textured_ubo_layout, None);

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

            if VALIDATION.is_enable {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_merssager, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
