use ash::vk;
use glam::{Mat4, Vec3};

use memoffset::offset_of;

/// World position used to "park" a registered instance off-screen.
///
/// `register_instance` always emits a draw, and `draw_frame` reuses the
/// instance's last transform if a fresh one isn't supplied. To hide an
/// instance we therefore translate it far enough that perspective culls it
/// rather than try to skip it from the draw list. Used by `Digit` (unlit
/// segments) and `TextLabel` (hidden labels). See `hidden_transform`.
pub const HIDDEN_TRANSLATION: Vec3 = Vec3::new(1000.0, 1000.0, 0.0);

/// Convenience for the "park off-screen" pattern. Equivalent to
/// `Mat4::from_translation(HIDDEN_TRANSLATION)`.
pub fn hidden_transform() -> Mat4 {
    Mat4::from_translation(HIDDEN_TRANSLATION)
}

pub struct DeviceExtension {
    pub names: [&'static str; 1],
    //    pub raw_names: [*const i8; 1],
}

pub struct SurfaceStuff {
    pub surface_loader: ash::extensions::khr::Surface,
    pub surface: vk::SurfaceKHR,
}
pub struct SwapChainStuff {
    pub swapchain_loader: ash::extensions::khr::Swapchain,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
}

pub struct SwapChainSupportDetail {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new() -> QueueFamilyIndices {
        QueueFamilyIndices {
            graphics_family: None,
            present_family: None,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

pub struct SyncObjects {
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub inflight_fences: Vec<vk::Fence>,
}

#[derive(Clone)]
pub struct ModelMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Build an axis-aligned rectangle mesh centred on the origin.
/// Used by every solid-colour drawable in the game (paddles, walls, digit segments).
pub fn rect_mesh(width: f32, height: f32) -> ModelMesh {
    let hw = width / 2.0;
    let hh = height / 2.0;
    ModelMesh {
        vertices: vec![
            Vertex { pos: [-hw, -hh] },
            Vertex { pos: [ hw, -hh] },
            Vertex { pos: [ hw,  hh] },
            Vertex { pos: [-hw,  hh] },
        ],
        indices: vec![0u32, 1, 2, 2, 3, 0],
    }
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct UniformBufferObject {
    pub view: Mat4,
    pub proj: Mat4,
}

// Solid-colour vertex: position only. Per-instance colour comes through
// `Instance` at binding 1, so meshes are reusable across instances of any
// colour. See `create_graphics_pipeline` for the two-binding vertex input.
#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: [f32; 2],
}
impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: ::std::mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 1] {
        [vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: offset_of!(Vertex, pos) as u32,
        }]
    }
}

// Per-instance data for the solid-colour pipeline. Layout must match the
// vertex shader (locations 1..4 = mat4 columns, location 5 = colour). glam's
// Mat4 is column-major with a 16-byte alignment, so the four columns map
// directly to four vec4 attributes at offsets 0/16/32/48.
#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct Instance {
    pub model: Mat4,
    pub color: [f32; 3],
}
impl Instance {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 1,
            stride: ::std::mem::size_of::<Instance>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        }
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 5] {
        [
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 16,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 48,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Instance, color) as u32,
            },
        ]
    }
}

// Textured vertex: position only on the shared unit quad ([-0.5..0.5]^2).
// UV is computed in the vertex shader from the per-instance UV rect, which
// lets every textured instance share one VBO/IBO.
#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct TexturedVertex {
    pub pos: [f32; 2],
}
impl TexturedVertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: ::std::mem::size_of::<TexturedVertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 1] {
        [vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: offset_of!(TexturedVertex, pos) as u32,
        }]
    }
}

// Per-instance data for the textured pipeline. Mat4 model at locations 1..4,
// uv_offset at 5, uv_scale at 6. The vertex shader maps unit-quad pos
// `[-0.5..0.5]^2` to `[0..1]^2` and computes `uv = uv_offset + unit * uv_scale`.
#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct TexturedInstance {
    pub model: Mat4,
    pub uv_offset: [f32; 2],
    pub uv_scale: [f32; 2],
}
impl TexturedInstance {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 1,
            stride: ::std::mem::size_of::<TexturedInstance>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        }
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 6] {
        [
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 2,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 16,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 3,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 48,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 5,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(TexturedInstance, uv_offset) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 6,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(TexturedInstance, uv_scale) as u32,
            },
        ]
    }
}
