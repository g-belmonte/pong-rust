//! Descriptor pool creation. Pool sizing is worst-case — every material is
//! assumed to be sampler-using since pool sizing happens before any material
//! is registered. See `create_descriptor_pool` for details.

use ash::vk;

use std::ptr;

pub fn create_descriptor_pool(
    device: &ash::Device,
    swapchain_images_size: usize,
    max_materials: usize,
    max_textured_models: usize,
) -> vk::DescriptorPool {
    // Sizing: one camera-only set per (non-sampler material, swapchain image)
    // PLUS up to `max_textured_models` sets per (sampler material, texture,
    // swapchain image). We can't tell at pool-creation time which materials
    // will be sampler-using and which won't, so we budget for the worst case:
    // every material registered as sampler-using.
    let per_material = max_textured_models * swapchain_images_size;
    let camera_only_sets = (max_materials * swapchain_images_size) as u32;
    let textured_sets = (max_materials * per_material) as u32;

    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: camera_only_sets + textured_sets,
            ..Default::default()
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: textured_sets,
            ..Default::default()
        },
    ];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        p_next: ptr::null(),
        // FREE_DESCRIPTOR_SET so unregister_texture can return per-(material,
        // texture) sets to the pool.
        flags: vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
        max_sets: camera_only_sets + textured_sets,
        pool_size_count: pool_sizes.len() as u32,
        p_pool_sizes: pool_sizes.as_ptr(),
        ..Default::default()
    };

    unsafe {
        device
            .create_descriptor_pool(&descriptor_pool_create_info, None)
            .expect("Failed to create Descriptor Pool!")
    }
}
