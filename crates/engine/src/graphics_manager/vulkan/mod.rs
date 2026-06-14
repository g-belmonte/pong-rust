//! Low-level Vulkan helpers used by [`GraphicsManager`](super::GraphicsManager).
//!
//! Each submodule covers one slice of the Vulkan API surface (init, swapchain,
//! buffers, images, render pass, commands, descriptors, shader). Items are
//! re-exported at the [`vk`](self) module root so callers can use them as
//! `vk::create_instance` etc. regardless of which submodule the function
//! actually lives in.

pub mod buffers;
pub mod commands;
pub mod descriptors;
pub mod images;
pub mod init;
pub mod render_pass;
pub mod shader;
pub mod swapchain;

pub use buffers::{
    copy_buffer, create_buffer, create_index_buffer, create_uniform_buffers, create_vertex_buffer,
    find_memory_type,
};
pub use commands::{
    allocate_command_buffers, begin_single_time_command, create_command_pool,
    create_render_finished_semaphores, create_sync_objects, end_single_time_command,
};
pub use descriptors::create_descriptor_pool;
pub use images::{
    create_depth_attachment, create_image_view, create_image_views, create_texture_sampler,
    load_texture_image, pick_depth_format, upload_rgba_image,
};
pub use init::{
    create_instance, create_logical_device, create_surface, pick_physical_device,
};
pub use render_pass::{create_framebuffers, create_render_pass};
pub use shader::create_shader_module;
pub use swapchain::create_swapchain;
