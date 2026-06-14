//! Swapchain teardown and recreation. Triggered on resize / `OUT_OF_DATE`.
//! Descriptor-set layouts, descriptor pool, all descriptor sets, the unit
//! quad, and per-material instance buffers survive across recreations.
//! Pipelines rebuild every recreation (viewport/scissor baked in).

use super::material::create_pipeline as create_material_pipeline;
use super::structures::SurfaceStuff;
use super::{vulkan, GraphicsManager};

impl GraphicsManager {
    pub(super) fn recreate_swapchain(&mut self) {
        let surface_suff = SurfaceStuff {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
        };

        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.cleanup_swapchain();

        let swapchain_stuff = vulkan::create_swapchain(
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
        let new_format = swapchain_stuff.swapchain_format;
        let format_changed = new_format != self.swapchain_format;
        self.swapchain_format = new_format;
        self.swapchain_extent = swapchain_stuff.swapchain_extent;

        let new_image_count = self.swapchain_images.len();
        if new_image_count != self.render_finished_semaphores.len() {
            unsafe {
                for &s in self.render_finished_semaphores.iter() {
                    self.device.destroy_semaphore(s, None);
                }
            }
            self.render_finished_semaphores =
                vulkan::create_render_finished_semaphores(&self.device, new_image_count);
        }

        // Invalidate every slot's cached matrices so the next per-slot
        // `set_camera` writes fresh UBOs against the new aspect.
        self.last_camera_views.clear();
        self.last_camera_projs.clear();

        self.swapchain_imageviews =
            vulkan::create_image_views(&self.device, self.swapchain_format, &self.swapchain_images);
        if format_changed {
            unsafe {
                self.device.destroy_render_pass(self.render_pass, None);
            }
            self.render_pass = vulkan::create_render_pass(
                &self.device,
                self.swapchain_format,
                self.depth_format,
            );
        }

        // Rebuild the depth attachment to match the new swapchain extent.
        unsafe {
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);
        }
        let (depth_image, depth_image_memory, depth_image_view) = vulkan::create_depth_attachment(
            &self.device,
            &self.physical_device_memory_properties,
            self.depth_format,
            self.swapchain_extent,
        );
        self.depth_image = depth_image;
        self.depth_image_memory = depth_image_memory;
        self.depth_image_view = depth_image_view;

        // Rebuild every material's pipeline (viewport/scissor are baked in,
        // so a swapchain-extent change invalidates them). Descriptor set
        // layouts and instance buffers survive — only the pipeline + pipeline
        // layout are recreated.
        for mat in self.materials.values_mut() {
            unsafe {
                self.device.destroy_pipeline(mat.pipeline, None);
                self.device
                    .destroy_pipeline_layout(mat.pipeline_layout, None);
            }
            let (pipeline, pipeline_layout, _vs, _is) = create_material_pipeline(
                &self.device,
                self.render_pass,
                self.swapchain_extent,
                mat.descriptor_set_layout,
                self.pipeline_cache,
                &mat.vertex_spv,
                &mat.fragment_spv,
                &mat.vertex_attrs,
                &mat.instance_attrs,
                mat.depth,
                mat.blend,
            );
            mat.pipeline = pipeline;
            mat.pipeline_layout = pipeline_layout;
        }

        self.swapchain_framebuffers = vulkan::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_imageviews,
            self.depth_image_view,
            self.swapchain_extent,
        );
    }

    pub(super) fn cleanup_swapchain(&self) {
        unsafe {
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for &image_view in self.swapchain_imageviews.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}
