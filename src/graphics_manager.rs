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
use constants::*;
use structures::{QueueFamilyIndices, SurfaceStuff};

use ash::version::DeviceV1_0;
use ash::version::InstanceV1_0;
use ash::vk;

use std::ptr;

use crate::scene::Scene;

use self::structures::UniformBufferObject;

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

pub struct GraphicsManager {
    window: winit::window::Window,

    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_merssager: vk::DebugUtilsMessengerEXT,

    physical_device: vk::PhysicalDevice,
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

    model_buffers: Vec<ModelBuffers>,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    is_framebuffer_resized: bool,
}

impl GraphicsManager {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>, scene: &Scene) -> GraphicsManager {
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

        let model_data = scene.get_model_data();
        let model_buffers: Vec<ModelBuffers> = model_data
            .iter()
            .map(|md| {
                let (vertex_buffer, vertex_buffer_memory) = share::create_vertex_buffer(
                    &device,
                    &physical_device_memory_properties,
                    command_pool,
                    graphics_queue,
                    &md.model_mesh.vertices,
                );
                let (index_buffer, index_buffer_memory) = share::create_index_buffer(
                    &device,
                    &physical_device_memory_properties,
                    command_pool,
                    graphics_queue,
                    &md.model_mesh.indices,
                );
                let (uniform_buffers, uniform_buffers_memory) = share::create_uniform_buffers(
                    &device,
                    &physical_device_memory_properties,
                    swapchain_stuff.swapchain_images.len(),
                );
                let descriptor_pool =
                    share::create_descriptor_pool(&device, swapchain_stuff.swapchain_images.len());
                let descriptor_sets = share::create_descriptor_sets(
                    &device,
                    descriptor_pool,
                    ubo_layout,
                    &uniform_buffers,
                    swapchain_stuff.swapchain_images.len(),
                );

                ModelBuffers {
                    vertex_buffer,
                    vertex_buffer_memory,
                    index_buffer,
                    index_buffer_memory,
                    index_count: md.model_mesh.indices.len() as u32,
                    uniform_transform: UniformBufferObject {
                        model: md.model_transform,
                        view: scene.camera.view,
                        proj: scene.camera.proj,
                    },
                    uniform_buffers,
                    uniform_buffers_memory,
                    descriptor_pool,
                    descriptor_sets,
                }
            })
            .collect();

        let command_buffers = share::create_command_buffers(
            &device,
            command_pool,
            graphics_pipeline,
            &swapchain_framebuffers,
            render_pass,
            swapchain_stuff.swapchain_extent,
            pipeline_layout,
            &model_buffers,
        );
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

    pub fn draw_frame(&mut self, transforms: Vec<Matrix4<f32>>) {
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

    fn update_uniform_buffer(&mut self, current_image: usize, transforms: Vec<Matrix4<f32>>) {
        for (i, buffers) in self.model_buffers.iter_mut().enumerate() {
            buffers.uniform_transform.model = transforms[i];
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
        for buffers in self.model_buffers.iter_mut() {
            buffers.uniform_transform = UniformBufferObject {
                model: buffers.uniform_transform.model,
                view: buffers.uniform_transform.view,
                proj: cgmath::perspective(
                    Deg(45.0),
                    self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
                    0.1,
                    10.0,
                ),
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
        self.command_buffers = share::create_command_buffers(
            &self.device,
            self.command_pool,
            self.graphics_pipeline,
            &self.swapchain_framebuffers,
            self.render_pass,
            self.swapchain_extent,
            pipeline_layout,
            &self.model_buffers,
        );
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

            for buffers in self.model_buffers.iter() {
                self.device
                    .destroy_descriptor_pool(buffers.descriptor_pool, None);

                for i in 0..buffers.uniform_buffers.len() {
                    self.device.destroy_buffer(buffers.uniform_buffers[i], None);
                    self.device
                        .free_memory(buffers.uniform_buffers_memory[i], None);
                }

                self.device.destroy_buffer(buffers.index_buffer, None);
                self.device.free_memory(buffers.index_buffer_memory, None);

                self.device.destroy_buffer(buffers.vertex_buffer, None);
                self.device.free_memory(buffers.vertex_buffer_memory, None);
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
