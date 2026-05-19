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

#[derive(Default)]
pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new() -> QueueFamilyIndices {
        QueueFamilyIndices::default()
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

/// Geometry buffer for one registered mesh. The vertex layout is owned by the
/// material that draws against the mesh, not by `ModelMesh` itself —
/// `vertex_bytes` is a raw packed buffer of per-vertex records and
/// `vertex_stride` is the byte size of one record. `indices` is always `u32`.
///
/// Pre-Phase-8 this carried a typed `Vec<Vertex>` (2D positions only). Making
/// it bytes lets the same registration path serve 2D quads, 3D cubes, lit
/// meshes with normals + UVs, and anything else a material declares — without
/// adding a parallel `Mesh3D` axis to every renderer API.
#[derive(Clone)]
pub struct ModelMesh {
    pub vertex_bytes: Vec<u8>,
    pub vertex_stride: u32,
    pub indices: Vec<u32>,
}

/// Build an axis-aligned 2D rectangle mesh centred on the origin. Vertex
/// layout is a single `vec2` position per vertex (8 bytes), matching the
/// built-in solid material's `vertex_attrs: [F32x2]`.
///
/// Used by every solid-colour drawable in Pong (paddles, walls, digit
/// segments).
pub fn rect_mesh(width: f32, height: f32) -> ModelMesh {
    let hw = width / 2.0;
    let hh = height / 2.0;
    let positions: [[f32; 2]; 4] = [
        [-hw, -hh],
        [ hw, -hh],
        [ hw,  hh],
        [-hw,  hh],
    ];
    let mut vertex_bytes = Vec::with_capacity(positions.len() * 8);
    for p in positions.iter() {
        vertex_bytes.extend_from_slice(unsafe {
            ::std::slice::from_raw_parts(p.as_ptr() as *const u8, 8)
        });
    }
    ModelMesh {
        vertex_bytes,
        vertex_stride: 8,
        indices: vec![0u32, 1, 2, 2, 3, 0],
    }
}

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct UniformBufferObject {
    pub view: Mat4,
    pub proj: Mat4,
}
