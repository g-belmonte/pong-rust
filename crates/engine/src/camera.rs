//! Cameras.
//!
//! [`Camera`] is the trait the renderer consumes. The active camera lives on
//! [`Scene::camera`](crate::scene::Scene::camera) as a `Box<dyn Camera>` —
//! game code swaps it in at scene-build time (e.g. `scene.camera =
//! Box::new(Camera3D::new(...))`). The renderer calls `view()` and
//! `proj(aspect)` once per frame via
//! [`GraphicsManager::set_camera`](crate::graphics_manager::GraphicsManager::set_camera).
//!
//! Two implementations ship today: [`Camera2D`] (orthographic, Y-down clip
//! space — matches Pong's convention) and [`Camera3D`] (perspective, Y-up
//! world with the clip-space flip done inside `proj`). Phase 8's "2D HUD on
//! top of a 3D world" idea will land as a per-material camera slot extension
//! to `Binding::CameraUbo`; today the engine writes a single `(view, proj)`
//! UBO that every material samples.

use glam::{Mat4, Vec2, Vec3};

/// Source of view + projection matrices the renderer uploads each frame.
pub trait Camera {
    /// World → view-space transform.
    fn view(&self) -> Mat4;
    /// View → Vulkan clip space, given the current `aspect = width / height`
    /// of the swapchain.
    fn proj(&self, aspect: f32) -> Mat4;
}

/// Orthographic 2D camera.
///
/// ## Convention
///
/// Positive Y is downwards (matches the scene + Vulkan clip space). The
/// projection produces Vulkan-style clip space directly: x ∈ [-1, 1],
/// y ∈ [-1, 1] with +y at the bottom of the screen, z ∈ [0, 1] over the
/// `[-1, 1]` world depth range.
///
/// ## Visible area
///
/// `half_height` is the world-space half-height of the visible area. The
/// half-width is derived from the renderer's current swapchain aspect ratio,
/// so resizing the window keeps `half_height` of vertical content visible and
/// reveals/hides horizontal content. This matches typical letterbox-vs-pillarbox
/// expectations for a 2D game with a fixed UI height.
#[derive(Clone, Copy, Debug)]
pub struct Camera2D {
    pub centre: Vec2,
    pub half_height: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            centre: Vec2::ZERO,
            half_height: 1.0,
        }
    }
}

impl Camera2D {
    pub fn new(centre: Vec2, half_height: f32) -> Self {
        Self { centre, half_height }
    }

    /// Convert a window-pixel coordinate (top-left origin, physical pixels —
    /// what [`crate::input::Input::mouse_position`] returns) into the
    /// world-space coordinate this camera draws there. `extent` is the
    /// window/swapchain size, available as
    /// [`crate::graphics_manager::GraphicsManager::extent`].
    ///
    /// Useful for mouse hit-testing against world-space UI: convert the
    /// cursor once per frame and intersect with each option's known
    /// world-space rect.
    pub fn screen_to_world(&self, px: f32, py: f32, extent: [u32; 2]) -> Vec2 {
        let w = extent[0] as f32;
        let h = extent[1] as f32;
        // Pixel → normalised device coords. Pixel y grows downwards;
        // Vulkan clip space y also grows downwards, so no flip.
        let ndc_x = (px / w) * 2.0 - 1.0;
        let ndc_y = (py / h) * 2.0 - 1.0;
        let aspect = w / h;
        let half_width = self.half_height * aspect;
        Vec2::new(
            self.centre.x + ndc_x * half_width,
            self.centre.y + ndc_y * self.half_height,
        )
    }
}

impl Camera for Camera2D {
    fn view(&self) -> Mat4 {
        Mat4::from_translation(Vec3::new(-self.centre.x, -self.centre.y, 0.0))
    }

    fn proj(&self, aspect: f32) -> Mat4 {
        let half_width = self.half_height * aspect;
        Mat4::from_cols_array(&[
            1.0 / half_width, 0.0,                    0.0, 0.0,
            0.0,              1.0 / self.half_height, 0.0, 0.0,
            0.0,              0.0,                    0.5, 0.0,
            0.0,              0.0,                    0.5, 1.0,
        ])
    }
}

/// Perspective 3D camera (right-handed look-at).
///
/// ## Convention
///
/// **World is Y-up.** `Camera3D::proj` flips Y in the projection so world +Y
/// (up) ends up at the top of the screen despite Vulkan's clip space having
/// +Y at the bottom. The depth range is Vulkan-native `[0, 1]` (glam's
/// `Mat4::perspective_rh`).
///
/// ## Fields
///
/// - `eye` / `target` / `up` feed `Mat4::look_at_rh`.
/// - `fov_y_radians` is the *vertical* field of view; the horizontal FOV is
///   derived from the swapchain aspect at render time.
/// - `near` / `far` are the depth bounds in view-space units; pick them tight
///   for the scene's content range to keep depth precision usable.
#[derive(Clone, Copy, Debug)]
pub struct Camera3D {
    pub eye: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub fov_y_radians: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            eye: Vec3::new(0.0, 0.0, 3.0),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y_radians: std::f32::consts::FRAC_PI_3,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera3D {
    pub fn new(eye: Vec3, target: Vec3, up: Vec3, fov_y_radians: f32, near: f32, far: f32) -> Self {
        Self { eye, target, up, fov_y_radians, near, far }
    }
}

impl Camera for Camera3D {
    fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye, self.target, self.up)
    }

    fn proj(&self, aspect: f32) -> Mat4 {
        // glam's `perspective_rh` returns a Vulkan-style projection with
        // depth ∈ [0, 1]. We still need to flip Y because Vulkan clip space
        // has +Y at the *bottom* of the screen while the world is Y-up.
        let mut p = Mat4::perspective_rh(self.fov_y_radians, aspect, self.near, self.far);
        // Column-major: column index 1 is the Y axis.
        let y_axis = p.y_axis;
        p.y_axis = -y_axis;
        p
    }
}
