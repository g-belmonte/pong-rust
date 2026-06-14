//! Image creation, layout transitions, views, samplers, and the depth attachment.

use ash::vk;

use std::ptr;

use super::buffers::{create_buffer, find_memory_type};
use super::commands::{begin_single_time_command, end_single_time_command};

/// Pick a supported depth format with `DEPTH_STENCIL_ATTACHMENT` optimal-tiling
/// support. Prefers `D32_SFLOAT` (depth-only, widely supported, no stencil
/// overhead); falls through to the stencil-bearing variants if needed.
pub fn pick_depth_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::Format {
    let candidates = [
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    for &f in candidates.iter() {
        let props =
            unsafe { instance.get_physical_device_format_properties(physical_device, f) };
        if props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return f;
        }
    }
    panic!("No supported depth attachment format")
}

/// Create a single depth attachment (image + memory + image view) sized to
/// `extent`. One depth attachment is shared across all swapchain framebuffers;
/// the render pass's external→subpass-0 dependency below serialises
/// depth-attachment writes across frames in flight.
pub fn create_depth_attachment(
    device: &ash::Device,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    format: vk::Format,
    extent: vk::Extent2D,
) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
    let image_create_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ImageCreateFlags::empty(),
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D { width: extent.width, height: extent.height, depth: 1 },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };
    let image = unsafe {
        device
            .create_image(&image_create_info, None)
            .expect("Failed to create depth image")
    };
    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
    let memory_type = find_memory_type(
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        device_memory_properties,
    );
    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),
        allocation_size: mem_requirements.size,
        memory_type_index: memory_type,
        ..Default::default()
    };
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocate depth image memory")
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("Failed to bind depth image memory");
    }
    let view = create_image_view(device, image, format, vk::ImageAspectFlags::DEPTH, 1);
    (image, memory, view)
}

// Decode `png_bytes`, stage to a buffer, create a device-local vk::Image, copy
// staging→image inside a single-time command, and transition the image into
// SHADER_READ_ONLY_OPTIMAL ready for sampling.
pub fn load_texture_image(
    device: &ash::Device,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    png_bytes: &[u8],
) -> (vk::Image, vk::DeviceMemory, u32, u32) {
    let decoded = image::load_from_memory(png_bytes)
        .expect("Failed to decode texture PNG");
    let rgba = decoded.to_rgba8();
    let (width, height) = (rgba.width(), rgba.height());
    let pixels = rgba.into_raw();
    let (image, memory) = upload_rgba_image(
        device,
        device_memory_properties,
        command_pool,
        submit_queue,
        width,
        height,
        &pixels,
    );
    (image, memory, width, height)
}

pub fn upload_rgba_image(
    device: &ash::Device,
    device_memory_properties: &vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    width: u32,
    height: u32,
    pixels: &[u8],
) -> (vk::Image, vk::DeviceMemory) {
    assert_eq!(
        pixels.len(),
        (width as usize) * (height as usize) * 4,
        "upload_rgba_image: pixels length must be width*height*4"
    );
    let image_size = pixels.len() as vk::DeviceSize;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        device,
        image_size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        device_memory_properties,
    );
    unsafe {
        let data_ptr = device
            .map_memory(staging_buffer_memory, 0, image_size, vk::MemoryMapFlags::empty())
            .expect("Failed to Map texture staging memory") as *mut u8;
        data_ptr.copy_from_nonoverlapping(pixels.as_ptr(), pixels.len());
        device.unmap_memory(staging_buffer_memory);
    }

    let image_create_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ImageCreateFlags::empty(),
        image_type: vk::ImageType::TYPE_2D,
        format: vk::Format::R8G8B8A8_SRGB,
        extent: vk::Extent3D { width, height, depth: 1 },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        queue_family_index_count: 0,
        p_queue_family_indices: ptr::null(),
        initial_layout: vk::ImageLayout::UNDEFINED,
        ..Default::default()
    };

    let texture_image = unsafe {
        device
            .create_image(&image_create_info, None)
            .expect("Failed to create texture image")
    };

    let mem_requirements = unsafe { device.get_image_memory_requirements(texture_image) };
    let memory_type = find_memory_type(
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
        device_memory_properties,
    );
    let alloc_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        p_next: ptr::null(),
        allocation_size: mem_requirements.size,
        memory_type_index: memory_type,
        ..Default::default()
    };
    let texture_image_memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocate texture image memory")
    };
    unsafe {
        device
            .bind_image_memory(texture_image, texture_image_memory, 0)
            .expect("Failed to bind texture image memory");
    }

    transition_image_layout(
        device,
        command_pool,
        submit_queue,
        texture_image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );
    copy_buffer_to_image(
        device,
        command_pool,
        submit_queue,
        staging_buffer,
        texture_image,
        width,
        height,
    );
    transition_image_layout(
        device,
        command_pool,
        submit_queue,
        texture_image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    (texture_image, texture_image_memory)
}

fn transition_image_layout(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let command_buffer = begin_single_time_command(device, command_pool);

    let (src_access_mask, dst_access_mask, src_stage, dst_stage) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => panic!("Unsupported image layout transition: {:?} -> {:?}", old_layout, new_layout),
        };

    let barrier = vk::ImageMemoryBarrier {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
        p_next: ptr::null(),
        src_access_mask,
        dst_access_mask,
        old_layout,
        new_layout,
        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        },
        ..Default::default()
    };

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }

    end_single_time_command(device, command_pool, submit_queue, command_buffer);
}

fn copy_buffer_to_image(
    device: &ash::Device,
    command_pool: vk::CommandPool,
    submit_queue: vk::Queue,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) {
    let command_buffer = begin_single_time_command(device, command_pool);

    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D { width, height, depth: 1 },
        ..Default::default()
    };

    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
    }

    end_single_time_command(device, command_pool, submit_queue, command_buffer);
}

pub fn create_texture_sampler(device: &ash::Device) -> vk::Sampler {
    let sampler_create_info = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::SamplerCreateFlags::empty(),
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::REPEAT,
        mip_lod_bias: 0.0,
        // Anisotropy disabled — the per-CLAUDE.md plan keeps sampler simple.
        // The device feature is enabled but we don't request anisotropy here.
        anisotropy_enable: vk::FALSE,
        max_anisotropy: 1.0,
        compare_enable: vk::FALSE,
        compare_op: vk::CompareOp::ALWAYS,
        min_lod: 0.0,
        max_lod: 0.0,
        border_color: vk::BorderColor::INT_OPAQUE_BLACK,
        unnormalized_coordinates: vk::FALSE,
        ..Default::default()
    };
    unsafe {
        device
            .create_sampler(&sampler_create_info, None)
            .expect("Failed to create texture sampler")
    }
}

pub fn create_image_views(
    device: &ash::Device,
    surface_format: vk::Format,
    images: &[vk::Image],
) -> Vec<vk::ImageView> {
    let swapchain_imageviews: Vec<vk::ImageView> = images
        .iter()
        .map(|&image| {
            create_image_view(
                device,
                image,
                surface_format,
                vk::ImageAspectFlags::COLOR,
                1,
            )
        })
        .collect();

    swapchain_imageviews
}

pub fn create_image_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect_flags: vk::ImageAspectFlags,
    mip_levels: u32,
) -> vk::ImageView {
    let imageview_create_info = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::ImageViewCreateFlags::empty(),
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        components: vk::ComponentMapping {
            r: vk::ComponentSwizzle::IDENTITY,
            g: vk::ComponentSwizzle::IDENTITY,
            b: vk::ComponentSwizzle::IDENTITY,
            a: vk::ComponentSwizzle::IDENTITY,
        },
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: aspect_flags,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        },
        image,
        ..Default::default()
    };

    unsafe {
        device
            .create_image_view(&imageview_create_info, None)
            .expect("Failed to create Image View!")
    }
}
