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

use self::structures::{ModelMesh, UniformBufferObject};

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ModelHandle(u32);

pub struct ModelBuffers {
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    index_count: u32,

    uniform_transform: UniformBufferObject,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

fn build_model_buffers(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    mesh: &ModelMesh,
    swapchain_image_count: usize,
    ubo_layout: vk::DescriptorSetLayout,
) -> ModelBuffers {
    let (vertex_buffer, vertex_buffer_memory) =
        share::create_vertex_buffer(device, mem_props, command_pool, queue, &mesh.vertices);
    let (index_buffer, index_buffer_memory) =
        share::create_index_buffer(device, mem_props, command_pool, queue, &mesh.indices);
    let (uniform_buffers, uniform_buffers_memory) =
        share::create_uniform_buffers(device, mem_props, swapchain_image_count);
    let descriptor_pool = share::create_descriptor_pool(device, swapchain_image_count);
    let descriptor_sets = share::create_descriptor_sets(
        device,
        descriptor_pool,
        ubo_layout,
        &uniform_buffers,
        swapchain_image_count,
    );

    ModelBuffers {
        vertex_buffer,
        vertex_buffer_memory,
        index_buffer,
        index_buffer_memory,
        index_count: mesh.indices.len() as u32,
        uniform_transform: UniformBufferObject {
            model: Matrix4::from_scale(1.0),
            view: Matrix4::from_scale(1.0),
            proj: Matrix4::from_scale(1.0),
        },
        uniform_buffers,
        uniform_buffers_memory,
        descriptor_pool,
        descriptor_sets,
    }
}

fn destroy_model_buffers(device: &ash::Device, buffers: &ModelBuffers) {
    unsafe {
        device.destroy_descriptor_pool(buffers.descriptor_pool, None);

        for i in 0..buffers.uniform_buffers.len() {
            device.destroy_buffer(buffers.uniform_buffers[i], None);
            device.free_memory(buffers.uniform_buffers_memory[i], None);
        }

        device.destroy_buffer(buffers.index_buffer, None);
        device.free_memory(buffers.index_buffer_memory, None);

        device.destroy_buffer(buffers.vertex_buffer, None);
        device.free_memory(buffers.vertex_buffer_memory, None);
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

    model_buffers: HashMap<ModelHandle, ModelBuffers>,
    next_handle: u32,
    command_buffers_dirty: bool,
    current_view: Matrix4<f32>,
    current_proj: Matrix4<f32>,

    command_pool: vk::CommandPool,
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
        let surface_stuff =
            share::create_surface(&entry, &instance, &window, WINDOW_WIDTH, WINDOW_HEIGHT);
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
        let (graphics_pipeline, pipeline_layout) = share::create_graphics_pipeline(
            &device,
            render_pass,
            swapchain_stuff.swapchain_extent,
            ubo_layout,
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

        let model_buffers: HashMap<ModelHandle, ModelBuffers> = HashMap::new();
        let next_handle: u32 = 0;

        // Command buffers are built on the first draw_frame via the dirty flag — Scene hasn't
        // registered any models yet, so there's nothing meaningful to record here.
        let command_buffers: Vec<vk::CommandBuffer> = Vec::new();
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

            model_buffers,
            next_handle,
            command_buffers_dirty: true,
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
        let mut buffers = build_model_buffers(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            mesh,
            self.swapchain_images.len(),
            self.ubo_layout,
        );
        // seed the UBO with the camera matrices captured at init.
        // model matrix gets overwritten on the first draw_frame call.
        buffers.uniform_transform = UniformBufferObject {
            model: Matrix4::from_scale(1.0),
            view: self.current_view,
            proj: self.current_proj,
        };
        self.model_buffers.insert(handle, buffers);
        self.command_buffers_dirty = true;
        handle
    }

    // Sorted by ModelHandle ordinal so registration order = draw order.
    // The render pass has no depth attachment and the pipeline has no alpha blending,
    // so overdraw correctness depends on this stable order — do not switch to a HashSet.
    fn sorted_model_buffers(&self) -> Vec<&ModelBuffers> {
        let mut entries: Vec<_> = self.model_buffers.iter().collect();
        entries.sort_by_key(|(handle, _)| **handle);
        entries.into_iter().map(|(_, buffers)| buffers).collect()
    }

    pub fn unregister_model(&mut self, handle: ModelHandle) {
        if let Some(buffers) = self.model_buffers.remove(&handle) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_model");
            }
            destroy_model_buffers(&self.device, &buffers);
            self.command_buffers_dirty = true;
        }
    }

    fn rebuild_command_buffers(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("device_wait_idle failed in rebuild_command_buffers");
        }
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
        }
        self.command_buffers = share::create_command_buffers(
            &self.device,
            self.command_pool,
            self.graphics_pipeline,
            &self.swapchain_framebuffers,
            self.render_pass,
            self.swapchain_extent,
            self.pipeline_layout,
            &self.sorted_model_buffers(),
        );
        self.command_buffers_dirty = false;
    }

    pub fn draw_frame(&mut self, transforms: &[(ModelHandle, Matrix4<f32>)]) {
        if self.command_buffers_dirty {
            self.rebuild_command_buffers();
        }

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

        self.update_uniform_buffer(image_index as usize, transforms);

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
            p_command_buffers: &self.command_buffers[image_index as usize],
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

    // A registered model whose handle is missing from `transforms` keeps last frame's UBO
    // (lets a caller skip an update). A handle in `transforms` that isn't registered is a
    // programming error → panic.
    fn update_uniform_buffer(
        &mut self,
        current_image: usize,
        transforms: &[(ModelHandle, Matrix4<f32>)],
    ) {
        for (handle, transform) in transforms {
            let buffers = self
                .model_buffers
                .get_mut(handle)
                .expect("draw_frame received transform for unknown ModelHandle");
            buffers.uniform_transform.model = *transform;
            let ubos = [buffers.uniform_transform];
            let buffer_size = (std::mem::size_of::<UniformBufferObject>() * ubos.len()) as u64;

            unsafe {
                let data_ptr = self
                    .device
                    .map_memory(
                        buffers.uniform_buffers_memory[current_image],
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to Map Memory")
                    as *mut UniformBufferObject;

                data_ptr.copy_from_nonoverlapping(ubos.as_ptr(), ubos.len());

                self.device
                    .unmap_memory(buffers.uniform_buffers_memory[current_image]);
            }
        }
    }

    fn recreate_swapchain(&mut self) {
        // parameters -------------
        let surface_suff = SurfaceStuff {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
            screen_width: WINDOW_WIDTH,
            screen_height: WINDOW_HEIGHT,
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

        // update camera aspect ratio
        self.current_proj = cgmath::perspective(
            Deg(45.0),
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            0.1,
            10.0,
        );
        for buffers in self.model_buffers.values_mut() {
            buffers.uniform_transform = UniformBufferObject {
                model: buffers.uniform_transform.model,
                view: buffers.uniform_transform.view,
                proj: self.current_proj,
            }
        }

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

        self.swapchain_framebuffers = share::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_imageviews,
            self.swapchain_extent,
        );
        self.rebuild_command_buffers();
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
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

            for buffers in self.model_buffers.values() {
                destroy_model_buffers(&self.device, buffers);
            }

            self.device
                .destroy_descriptor_set_layout(self.ubo_layout, None);

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
