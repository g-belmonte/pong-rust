//! Material / shader registry.
//!
//! A [`Material`] packages a vertex+fragment shader pair, a vertex+instance
//! attribute layout, and a descriptor binding set into a registered pipeline.
//! Game code calls [`Resources::load_material`](crate::resources::Resources::load_material)
//! (which forwards to [`GraphicsManager::register_material`]) to register a
//! custom material; the engine itself registers two built-in materials at
//! startup ("solid" and "textured") and exposes their handles via
//! [`GraphicsManager::solid_material`] / [`GraphicsManager::textured_material`].
//!
//! Per-instance data layout follows the convention that the **first instance
//! attribute is always [`VertexAttr::Mat4`]** (the model matrix at shader
//! locations 1..4). Anything that follows is material-specific and is supplied
//! per-instance as a `&[u8]` blob through
//! [`GraphicsManager::register_material_instance`].
//!
//! Batching strategy is chosen by whether the material's binding list includes
//! [`Binding::Sampler2d`]:
//!
//! - Without a sampler ("solid-like"): each instance brings its own
//!   [`MeshHandle`]; the renderer groups instances by mesh and emits one
//!   `vkCmdDrawIndexed` per mesh.
//! - With a sampler ("textured-like"): every instance shares the engine's
//!   built-in unit-quad mesh; instances are grouped by [`TextureHandle`] and
//!   one `vkCmdDrawIndexed` is emitted per texture batch.

use ash::vk;
use std::collections::HashMap;
use std::ffi::CString;
use std::ptr;

use crate::graphics_manager::share;
use crate::graphics_manager::structures::UniformBufferObject;
use crate::graphics_manager::TextureHandle;

/// Opaque identifier for a registered [`Material`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct MaterialHandle(pub(crate) u32);

/// Vertex / instance attribute element types.
///
/// Attribute locations are assigned sequentially: vertex attrs start at
/// location 0, instance attrs continue from there. A `Mat4` consumes four
/// consecutive `vec4` slots (the standard Vulkan layout for matrix vertex
/// inputs).
#[derive(Copy, Clone, Debug)]
pub enum VertexAttr {
    F32x2,
    F32x3,
    F32x4,
    Mat4,
}

impl VertexAttr {
    fn size(self) -> u32 {
        match self {
            VertexAttr::F32x2 => 8,
            VertexAttr::F32x3 => 12,
            VertexAttr::F32x4 => 16,
            VertexAttr::Mat4 => 64,
        }
    }

    fn format(self) -> vk::Format {
        match self {
            VertexAttr::F32x2 => vk::Format::R32G32_SFLOAT,
            VertexAttr::F32x3 => vk::Format::R32G32B32_SFLOAT,
            VertexAttr::F32x4 => vk::Format::R32G32B32A32_SFLOAT,
            VertexAttr::Mat4 => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

/// Descriptor set bindings a material may consume.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Binding {
    /// The camera `(view, proj)` UBO at descriptor binding 0, sampled from
    /// the camera slot given by the `u32`. Slot 0 is the conventional
    /// default; secondary slots (e.g. slot 1 for a 2D HUD camera over a 3D
    /// world camera in slot 0) are populated via
    /// [`Scene::set_camera`](crate::scene::Scene::set_camera). A material may
    /// declare at most one `CameraUbo` binding — only one camera UBO is bound
    /// per pipeline.
    CameraUbo(u32),
    /// A combined image sampler bound at descriptor binding 1. Presence in the
    /// binding list flips the material into "textured-like" batching (instances
    /// keyed by texture, shared unit-quad mesh).
    Sampler2d,
}

/// Per-material depth-buffer behaviour. The engine's render pass always has a
/// depth attachment; this enum controls how a material's pipeline interacts
/// with it.
///
/// - [`DepthMode::Disabled`] — no test, no write. The right choice for 2D UI /
///   HUD layers that should always paint on top, and for the engine's built-in
///   solid + textured materials (they predate depth and rely on registration
///   order for layering).
/// - [`DepthMode::ReadOnly`] — depth test on, write off. Suits 2.5D sprites
///   that should be occluded by 3D geometry but shouldn't themselves block
///   what's drawn after them.
/// - [`DepthMode::ReadWrite`] — depth test on, write on. The standard choice
///   for opaque 3D geometry: pixels behind already-drawn geometry are culled,
///   and this material's pixels in turn occlude later draws.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum DepthMode {
    #[default]
    Disabled,
    ReadOnly,
    ReadWrite,
}

/// Per-material colour-blend behaviour.
///
/// - [`BlendMode::Opaque`] — blending disabled. The fragment shader's output
///   overwrites the framebuffer. The right choice for opaque geometry and for
///   any material that already handles transparency via `discard` (the
///   engine's textured material and pong's `text_tint` use alpha-discard).
/// - [`BlendMode::Alpha`] — straight (non-premultiplied) alpha blending:
///   `out = src.rgb * src.a + dst.rgb * (1 - src.a)`. Lets a material render
///   semi-transparent fragments (pause-menu dimmer, fade-outs).
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum BlendMode {
    #[default]
    Opaque,
    Alpha,
}

/// Material description supplied by game code (or by the engine for its
/// built-ins). SPIR-V bytes are copied at registration time, so callers may
/// `include_bytes!` from a `&'static [u8]` source.
pub struct MaterialDesc<'a> {
    pub vertex_spv: &'a [u8],
    pub fragment_spv: &'a [u8],
    /// Per-vertex attributes at binding 0. For the engine's built-in solid +
    /// textured materials this is `[F32x2]` (a 2D position).
    pub vertex_attrs: &'a [VertexAttr],
    /// Per-instance attributes at binding 1. **The first attribute must be
    /// [`VertexAttr::Mat4`]** — the engine writes the model matrix there.
    pub instance_attrs: &'a [VertexAttr],
    pub bindings: &'a [Binding],
    /// Depth-buffer interaction. Defaults to [`DepthMode::Disabled`] so the
    /// engine's built-in 2D materials (and any custom material that doesn't
    /// care about depth) keep paint-order layering. 3D opaque materials want
    /// [`DepthMode::ReadWrite`].
    pub depth: DepthMode,
    /// Colour-blend behaviour. Defaults to [`BlendMode::Opaque`] — the engine's
    /// existing materials all rely on alpha-discard in the fragment shader or
    /// are fully opaque. Pick [`BlendMode::Alpha`] for materials that need real
    /// transparency (semi-transparent UI overlays, fade effects).
    pub blend: BlendMode,
}

/// Internal registered material. Fields marked "rebuilt on swapchain
/// recreation" are torn down and recreated in `recreate_swapchain`; everything
/// else survives.
pub(crate) struct Material {
    // ----- Description (kept for pipeline rebuild on swapchain recreation) -----
    pub(crate) vertex_spv: Vec<u8>,
    pub(crate) fragment_spv: Vec<u8>,
    pub(crate) vertex_attrs: Vec<VertexAttr>,
    pub(crate) instance_attrs: Vec<VertexAttr>,

    // ----- Derived stride (bytes) -----
    pub(crate) instance_stride: u32,
    /// `instance_stride - 64` — bytes the caller supplies per instance after
    /// the model matrix (which is always the first 64 bytes).
    pub(crate) extra_size: u32,

    /// Depth-buffer behaviour. Threaded into `create_pipeline` on both
    /// initial registration and `recreate_swapchain` rebuilds.
    pub(crate) depth: DepthMode,

    /// Colour-blend behaviour. Threaded into `create_pipeline` on both initial
    /// registration and `recreate_swapchain` rebuilds.
    pub(crate) blend: BlendMode,

    /// Camera slot this material samples — extracted from `Binding::CameraUbo`
    /// at registration time. The renderer uses this to pick the right
    /// per-slot UBO buffers when allocating descriptor sets (initial +
    /// post-swapchain-recreation) and when writing camera matrices.
    pub(crate) camera_slot: u32,

    // ----- Pipeline state (rebuilt on swapchain recreation) -----
    pub(crate) descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) pipeline: vk::Pipeline,

    // ----- Per-frame instance buffer (survives recreation) -----
    pub(crate) instance_buffers: Vec<vk::Buffer>,
    pub(crate) instance_buffer_memories: Vec<vk::DeviceMemory>,

    /// `true` iff `bindings` contains [`Binding::Sampler2d`]. Controls the
    /// batching strategy: textured materials share the engine's unit quad and
    /// batch by texture; non-textured materials use per-instance meshes and
    /// batch by mesh.
    pub(crate) has_sampler: bool,

    /// For non-sampler materials only: camera-UBO descriptor set per swapchain
    /// image. Shared across every batch of this material.
    pub(crate) camera_descriptor_sets: Vec<vk::DescriptorSet>,

    /// For sampler materials only: per-texture descriptor sets (one set per
    /// swapchain image). Allocated lazily when an instance using texture T is
    /// first registered against this material, freed when the texture (or the
    /// material) is unregistered.
    pub(crate) texture_descriptor_sets: HashMap<TextureHandle, Vec<vk::DescriptorSet>>,
}

impl Material {
    pub(crate) fn has_sampler_binding(bindings: &[Binding]) -> bool {
        bindings.iter().any(|b| matches!(b, Binding::Sampler2d))
    }

    /// Extract the camera slot a material samples. Returns `None` for
    /// materials with no `CameraUbo` binding (none exist in the engine
    /// today, but the API permits it). Multiple `CameraUbo` bindings would
    /// hit the assert in [`crate::graphics_manager::GraphicsManager::register_material`].
    pub(crate) fn camera_slot(bindings: &[Binding]) -> Option<u32> {
        bindings.iter().find_map(|b| match b {
            Binding::CameraUbo(slot) => Some(*slot),
            _ => None,
        })
    }
}

/// Build the descriptor-set-layout bindings for a material, derived from its
/// [`Binding`] list. CameraUbo is bound to slot 0 in the vertex stage;
/// Sampler2d is bound to slot 1 in the fragment stage.
pub(crate) fn create_descriptor_set_layout(
    device: &ash::Device,
    bindings: &[Binding],
) -> vk::DescriptorSetLayout {
    let mut layout_bindings: Vec<vk::DescriptorSetLayoutBinding> = Vec::new();
    for b in bindings {
        match b {
            // The camera *slot* picks which UBO to bind from the renderer;
            // the descriptor-set binding index in the shader is always 0.
            Binding::CameraUbo(_slot) => layout_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                p_immutable_samplers: ptr::null(),
                ..Default::default()
            }),
            Binding::Sampler2d => layout_bindings.push(vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: ptr::null(),
                ..Default::default()
            }),
        }
    }
    let create_info = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
        binding_count: layout_bindings.len() as u32,
        p_bindings: layout_bindings.as_ptr(),
        ..Default::default()
    };
    unsafe {
        device
            .create_descriptor_set_layout(&create_info, None)
            .expect("Failed to create material descriptor set layout")
    }
}

/// Allocate per-swapchain-image descriptor sets for a non-sampler material —
/// each set binds just the camera UBO at binding 0.
pub(crate) fn allocate_camera_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    camera_uniform_buffers: &[vk::Buffer],
) -> Vec<vk::DescriptorSet> {
    let count = camera_uniform_buffers.len();
    let layouts = vec![layout; count];
    let allocate_info = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: ptr::null(),
        descriptor_pool: pool,
        descriptor_set_count: count as u32,
        p_set_layouts: layouts.as_ptr(),
        ..Default::default()
    };
    let sets = unsafe {
        device
            .allocate_descriptor_sets(&allocate_info)
            .expect("Failed to allocate material camera descriptor sets")
    };
    for (i, &set) in sets.iter().enumerate() {
        let buffer_info = [vk::DescriptorBufferInfo {
            buffer: camera_uniform_buffers[i],
            offset: 0,
            range: ::std::mem::size_of::<UniformBufferObject>() as u64,
            ..Default::default()
        }];
        let writes = [vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set: set,
            dst_binding: 0,
            dst_array_element: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: buffer_info.as_ptr(),
            p_texel_buffer_view: ptr::null(),
            ..Default::default()
        }];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
    sets
}

/// Allocate per-swapchain-image descriptor sets for a (material, texture) pair
/// — each set binds the camera UBO at binding 0 plus this texture's view +
/// sampler at binding 1.
pub(crate) fn allocate_textured_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    camera_uniform_buffers: &[vk::Buffer],
    image_view: vk::ImageView,
    sampler: vk::Sampler,
) -> Vec<vk::DescriptorSet> {
    let count = camera_uniform_buffers.len();
    let layouts = vec![layout; count];
    let allocate_info = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        p_next: ptr::null(),
        descriptor_pool: pool,
        descriptor_set_count: count as u32,
        p_set_layouts: layouts.as_ptr(),
        ..Default::default()
    };
    let sets = unsafe {
        device
            .allocate_descriptor_sets(&allocate_info)
            .expect("Failed to allocate material textured descriptor sets")
    };
    for (i, &set) in sets.iter().enumerate() {
        let buffer_info = [vk::DescriptorBufferInfo {
            buffer: camera_uniform_buffers[i],
            offset: 0,
            range: ::std::mem::size_of::<UniformBufferObject>() as u64,
            ..Default::default()
        }];
        let image_info = [vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ..Default::default()
        }];
        let writes = [
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: set,
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                p_image_info: ptr::null(),
                p_buffer_info: buffer_info.as_ptr(),
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
                p_next: ptr::null(),
                dst_set: set,
                dst_binding: 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: image_info.as_ptr(),
                p_buffer_info: ptr::null(),
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            },
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }
    sets
}

/// Compute `(stride, attribute_descriptions)` for a given binding +
/// starting-location pair, packing each attribute end-to-end. `Mat4` consumes
/// four locations (each a `vec4`) at consecutive 16-byte offsets.
pub(crate) fn build_attributes(
    binding: u32,
    start_location: u32,
    attrs: &[VertexAttr],
) -> (u32, u32, Vec<vk::VertexInputAttributeDescription>) {
    let mut offset: u32 = 0;
    let mut location = start_location;
    let mut out = Vec::new();
    for a in attrs {
        match a {
            VertexAttr::Mat4 => {
                for col in 0..4 {
                    out.push(vk::VertexInputAttributeDescription {
                        binding,
                        location: location + col,
                        format: vk::Format::R32G32B32A32_SFLOAT,
                        offset: offset + col * 16,
                    });
                }
                location += 4;
                offset += 64;
            }
            other => {
                out.push(vk::VertexInputAttributeDescription {
                    binding,
                    location,
                    format: other.format(),
                    offset,
                });
                location += 1;
                offset += other.size();
            }
        }
    }
    (offset, location - start_location, out)
}

/// Cap on simultaneously-live instances **per material**. Replaces the old
/// per-pipeline solid / textured instance caps that lived in `constants.rs`
/// before the Phase 7 material registry.
pub const MAX_INSTANCES_PER_MATERIAL: usize = 256;

/// Build the pipeline + pipeline-layout for a material. Shared by initial
/// registration and by `recreate_swapchain` (viewport/scissor are baked into
/// the pipeline, so it has to be rebuilt whenever the swapchain extent
/// changes).
#[allow(clippy::too_many_arguments)]
pub(crate) fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    swapchain_extent: vk::Extent2D,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_cache: vk::PipelineCache,
    vertex_spv: &[u8],
    fragment_spv: &[u8],
    vertex_attrs: &[VertexAttr],
    instance_attrs: &[VertexAttr],
    depth: DepthMode,
    blend: BlendMode,
) -> (vk::Pipeline, vk::PipelineLayout, u32, u32) {
    let vert_shader_module = share::create_shader_module(device, vertex_spv.to_vec());
    let frag_shader_module = share::create_shader_module(device, fragment_spv.to_vec());

    let main_function_name = CString::new("main").unwrap();

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            module: vert_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineShaderStageCreateFlags::empty(),
            module: frag_shader_module,
            p_name: main_function_name.as_ptr(),
            p_specialization_info: ptr::null(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        },
    ];

    let (vertex_stride, vertex_locs, vertex_attr_descs) = build_attributes(0, 0, vertex_attrs);
    let (instance_stride, _, instance_attr_descs) =
        build_attributes(1, vertex_locs, instance_attrs);

    let binding_descriptions = [
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: vertex_stride,
            input_rate: vk::VertexInputRate::VERTEX,
        },
        vk::VertexInputBindingDescription {
            binding: 1,
            stride: instance_stride,
            input_rate: vk::VertexInputRate::INSTANCE,
        },
    ];

    let mut attribute_descriptions: Vec<vk::VertexInputAttributeDescription> =
        Vec::with_capacity(vertex_attr_descs.len() + instance_attr_descs.len());
    attribute_descriptions.extend_from_slice(&vertex_attr_descs);
    attribute_descriptions.extend_from_slice(&instance_attr_descs);

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineVertexInputStateCreateFlags::empty(),
        vertex_attribute_description_count: attribute_descriptions.len() as u32,
        p_vertex_attribute_descriptions: attribute_descriptions.as_ptr(),
        vertex_binding_description_count: binding_descriptions.len() as u32,
        p_vertex_binding_descriptions: binding_descriptions.as_ptr(),
        ..Default::default()
    };
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
        p_next: ptr::null(),
        primitive_restart_enable: vk::FALSE,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: swapchain_extent.width as f32,
        height: swapchain_extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
        ..Default::default()
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: swapchain_extent,
        ..Default::default()
    }];
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineViewportStateCreateFlags::empty(),
        scissor_count: scissors.len() as u32,
        p_scissors: scissors.as_ptr(),
        viewport_count: viewports.len() as u32,
        p_viewports: viewports.as_ptr(),
        ..Default::default()
    };

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineRasterizationStateCreateFlags::empty(),
        depth_clamp_enable: vk::FALSE,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::CLOCKWISE,
        line_width: 1.0,
        polygon_mode: vk::PolygonMode::FILL,
        rasterizer_discard_enable: vk::FALSE,
        depth_bias_clamp: 0.0,
        depth_bias_constant_factor: 0.0,
        depth_bias_enable: vk::FALSE,
        depth_bias_slope_factor: 0.0,
        ..Default::default()
    };

    let multisample_state = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        flags: vk::PipelineMultisampleStateCreateFlags::empty(),
        p_next: ptr::null(),
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        sample_shading_enable: vk::FALSE,
        min_sample_shading: 0.0,
        p_sample_mask: ptr::null(),
        alpha_to_one_enable: vk::FALSE,
        alpha_to_coverage_enable: vk::FALSE,
        ..Default::default()
    };

    let stencil_state = vk::StencilOpState {
        fail_op: vk::StencilOp::KEEP,
        pass_op: vk::StencilOp::KEEP,
        depth_fail_op: vk::StencilOp::KEEP,
        compare_op: vk::CompareOp::ALWAYS,
        compare_mask: 0,
        write_mask: 0,
        reference: 0,
        ..Default::default()
    };
    let (depth_test, depth_write) = match depth {
        DepthMode::Disabled => (vk::FALSE, vk::FALSE),
        DepthMode::ReadOnly => (vk::TRUE, vk::FALSE),
        DepthMode::ReadWrite => (vk::TRUE, vk::TRUE),
    };
    let depth_state = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
        depth_test_enable: depth_test,
        depth_write_enable: depth_write,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        depth_bounds_test_enable: vk::FALSE,
        stencil_test_enable: vk::FALSE,
        front: stencil_state,
        back: stencil_state,
        max_depth_bounds: 1.0,
        min_depth_bounds: 0.0,
        ..Default::default()
    };

    // Per-material blend setup. Opaque is the historical default (no blend,
    // src overwrites dst). Alpha is straight (non-premultiplied) source-over —
    // see `BlendMode::Alpha` docs for the formula.
    let color_blend_attachments = [match blend {
        BlendMode::Opaque => vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
            ..Default::default()
        },
        BlendMode::Alpha => vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::TRUE,
            color_write_mask: vk::ColorComponentFlags::RGBA,
            src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
            dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            alpha_blend_op: vk::BlendOp::ADD,
            ..Default::default()
        },
    }];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineColorBlendStateCreateFlags::empty(),
        logic_op_enable: vk::FALSE,
        logic_op: vk::LogicOp::COPY,
        attachment_count: color_blend_attachments.len() as u32,
        p_attachments: color_blend_attachments.as_ptr(),
        blend_constants: [0.0, 0.0, 0.0, 0.0],
        ..Default::default()
    };

    let set_layouts = [descriptor_set_layout];
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineLayoutCreateFlags::empty(),
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
        ..Default::default()
    };
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&pipeline_layout_create_info, None)
            .expect("Failed to create material pipeline layout")
    };

    let pipeline_create_infos = [vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::PipelineCreateFlags::empty(),
        stage_count: shader_stages.len() as u32,
        p_stages: shader_stages.as_ptr(),
        p_vertex_input_state: &vertex_input_state,
        p_input_assembly_state: &input_assembly,
        p_tessellation_state: ptr::null(),
        p_viewport_state: &viewport_state,
        p_rasterization_state: &rasterization_state,
        p_multisample_state: &multisample_state,
        p_depth_stencil_state: &depth_state,
        p_color_blend_state: &color_blend_state,
        p_dynamic_state: ptr::null(),
        layout: pipeline_layout,
        render_pass,
        subpass: 0,
        base_pipeline_handle: vk::Pipeline::null(),
        base_pipeline_index: -1,
        ..Default::default()
    }];
    let pipelines = unsafe {
        device
            .create_graphics_pipelines(pipeline_cache, &pipeline_create_infos, None)
            .expect("Failed to create material graphics pipeline")
    };
    unsafe {
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
    }
    (pipelines[0], pipeline_layout, vertex_stride, instance_stride)
}
