//! Command pool, command-buffer allocation, single-time-command helpers, and
//! per-frame / per-image sync objects.

use ash::vk;

use std::ptr;

use crate::graphics_manager::structures::*;

pub fn create_command_pool(
    device: &ash::Device,
    queue_families: &QueueFamilyIndices,
) -> vk::CommandPool {
    let command_pool_create_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        p_next: ptr::null(),
        // Per-frame re-recording resets individual buffers via vkResetCommandBuffer.
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        queue_family_index: queue_families.graphics_family.unwrap(),
        ..Default::default()
    };

    unsafe {
        device
            .create_command_pool(&command_pool_create_info, None)
            .expect("Failed to create Command Pool!")
    }
}

pub fn allocate_command_buffers(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    count: u32,
) -> Vec<vk::CommandBuffer> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_buffer_count: count,
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        ..Default::default()
    };

    unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .expect("Failed to allocate Command Buffers!")
    }
}

pub fn begin_single_time_command(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> vk::CommandBuffer {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        p_next: ptr::null(),
        command_buffer_count: 1,
        command_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        ..Default::default()
    };

    let command_buffer = unsafe {
        device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .expect("Failed to allocate Command Buffers!")
    }[0];

    let command_buffer_begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        p_inheritance_info: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    unsafe {
        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Failed to begin recording Command Buffer at beginning!");
    }

    command_buffer
}

pub fn end_single_time_command(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
) {
    unsafe {
        device
            .end_command_buffer(command_buffer)
            .expect("Failed to record Command Buffer at Ending!");
    }

    let buffers_to_submit = [command_buffer];

    let sumbit_infos = [vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        p_next: ptr::null(),
        wait_semaphore_count: 0,
        p_wait_semaphores: ptr::null(),
        p_wait_dst_stage_mask: ptr::null(),
        command_buffer_count: 1,
        p_command_buffers: buffers_to_submit.as_ptr(),
        signal_semaphore_count: 0,
        p_signal_semaphores: ptr::null(),
        ..Default::default()
    }];

    unsafe {
        device
            .queue_submit(submit_queue, &sumbit_infos, vk::Fence::null())
            .expect("Failed to Queue Submit!");
        device
            .queue_wait_idle(submit_queue)
            .expect("Failed to wait Queue idle!");
        device.free_command_buffers(command_pool, &buffers_to_submit);
    }
}

// `image_available_semaphores` and `inflight_fences` are per frame-in-flight.
// `render_finished_semaphores` are per swapchain image — presentation reuse is
// governed by swapchain image index, so signaling one of N per-frame semaphores
// while the previous present on the same image is still in flight trips
// VUID-vkQueueSubmit-pSignalSemaphores-00067.
pub fn create_sync_objects(
    device: &ash::Device,
    max_frame_in_flight: usize,
    swapchain_image_count: usize,
) -> SyncObjects {
    let mut sync_objects = SyncObjects {
        image_available_semaphores: vec![],
        render_finished_semaphores: vec![],
        inflight_fences: vec![],
    };

    let semaphore_create_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
        ..Default::default()
    };

    let fence_create_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };

    for _ in 0..max_frame_in_flight {
        unsafe {
            let image_available_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .expect("Failed to create Semaphore Object!");
            let inflight_fence = device
                .create_fence(&fence_create_info, None)
                .expect("Failed to create Fence Object!");

            sync_objects
                .image_available_semaphores
                .push(image_available_semaphore);
            sync_objects.inflight_fences.push(inflight_fence);
        }
    }

    for _ in 0..swapchain_image_count {
        unsafe {
            let render_finished_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .expect("Failed to create Semaphore Object!");
            sync_objects
                .render_finished_semaphores
                .push(render_finished_semaphore);
        }
    }

    sync_objects
}

pub fn create_render_finished_semaphores(
    device: &ash::Device,
    count: usize,
) -> Vec<vk::Semaphore> {
    let semaphore_create_info = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SemaphoreCreateFlags::empty(),
        ..Default::default()
    };
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        unsafe {
            out.push(
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("Failed to create Semaphore Object!"),
            );
        }
    }
    out
}
