use ash::vk;
use glam::{Mat4, Vec3};

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
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
}
pub struct SwapChainStuff {
    pub swapchain_loader: ash::khr::swapchain::Device,
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

// 2D-position vertex used by [`ModelMesh`] (built-in solid material's
// per-vertex layout) and by the engine's shared unit-quad VBO (built-in
// textured material's mesh). Both built-ins declare `vertex_attrs: [F32x2]`
// in their [`MaterialDesc`](super::material::MaterialDesc), so the renderer
// derives the matching binding/attribute descriptions from there — `Vertex`
// itself no longer carries any descriptor-building methods.
#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct Vertex {
    pub pos: [f32; 2],
}

/// Alias for the engine's unit-quad VBO. Identical to [`Vertex`]; kept as a
/// distinct name because the unit quad serves the textured pipeline
/// specifically.
pub type TexturedVertex = Vertex;
