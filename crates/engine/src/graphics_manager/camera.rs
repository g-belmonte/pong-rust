//! Per-slot camera UBO management. Each material declares a slot via
//! [`Binding::CameraUbo(slot)`](super::Binding::CameraUbo); `set_camera` pushes
//! that slot's matrices into the UBOs. Cache-checked so a static camera in any
//! slot only pays the `device_wait_idle` cost the first time it changes.

use ash::vk;
use glam::Mat4;

use super::structures::UniformBufferObject;
use super::vulkan;
use super::GraphicsManager;
use crate::camera::Camera;

pub(super) fn write_camera_ubos(
    device: &ash::Device,
    memories: &[vk::DeviceMemory],
    view: Mat4,
    proj: Mat4,
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

impl GraphicsManager {
    /// Lazily allocate the per-swapchain-image UBO buffers for the given
    /// camera slot, plus a per-slot empty cache entry. Idempotent: returns
    /// immediately if the slot already exists. Called from `register_material`
    /// (for the slot the material samples) and from `set_camera` (for any
    /// slot a scene populates that no material samples yet).
    pub(super) fn ensure_camera_slot(&mut self, slot: u32) {
        if self.camera_uniform_buffers.contains_key(&slot) {
            return;
        }
        let swapchain_image_count = self.swapchain_images.len();
        let (buffers, memories) = vulkan::create_uniform_buffers(
            &self.device,
            &self.physical_device_memory_properties,
            swapchain_image_count,
        );
        self.camera_uniform_buffers.insert(slot, buffers);
        self.camera_uniform_buffers_memory.insert(slot, memories);
    }

    /// Push `camera`'s matrices into the UBOs for the given slot. The App
    /// calls this once per populated `Scene::cameras` slot per frame.
    /// Cache-checked per slot, so a static camera in any slot only pays the
    /// `device_wait_idle` cost the first time it changes.
    pub fn set_camera(&mut self, slot: u32, camera: &dyn Camera) {
        // A scene might set_camera a slot that no material samples (e.g. a
        // future toggle between cameras). Allocate UBOs for it so the write
        // below has somewhere to land.
        self.ensure_camera_slot(slot);

        let aspect =
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32;
        let view = camera.view();
        let proj = camera.proj(aspect);
        if self.last_camera_views.get(&slot) == Some(&view)
            && self.last_camera_projs.get(&slot) == Some(&proj)
        {
            return;
        }
        unsafe {
            self.device
                .device_wait_idle()
                .expect("device_wait_idle failed in set_camera");
        }
        write_camera_ubos(
            &self.device,
            &self.camera_uniform_buffers_memory[&slot],
            view,
            proj,
        );
        self.last_camera_views.insert(slot, view);
        self.last_camera_projs.insert(slot, proj);
    }
}
