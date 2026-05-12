//! 2D orthographic camera.
//!
//! [`Camera2D`] is the only camera type today. It lives on [`Scene::camera`]
//! (a top-level field, not an object component) and the renderer reads its
//! view/proj each frame via [`crate::graphics_manager::GraphicsManager::set_camera`].
//!
//! ## Convention
//!
//! Positive Y is downwards (matches the scene + Vulkan clip space). The
//! projection produces Vulkan-style clip space directly: x ∈ [-1, 1],
//! y ∈ [-1, 1] with +y at the bottom of the screen, z ∈ [0, 1] over the
//! `[-1, 1]` world depth range.
//!
//! ## Visible area
//!
//! `half_height` is the world-space half-height of the visible area. The
//! half-width is derived from the renderer's current swapchain aspect ratio,
//! so resizing the window keeps `half_height` of vertical content visible and
//! reveals/hides horizontal content. This matches typical letterbox-vs-pillarbox
//! expectations for a 2D game with a fixed UI height.
//!
//! ## Future 3D
//!
//! A `Camera3D` will land alongside `Camera2D` in Phase 8 when a 3D game
//! exists in the workspace. A trait abstraction isn't worth introducing
//! preemptively — see ARCHITECTURE.md's Phase 4 decision log.

use cgmath::{Matrix4, Vector2, Vector3};

#[derive(Clone, Copy, Debug)]
pub struct Camera2D {
    pub centre: Vector2<f32>,
    pub half_height: f32,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            centre: Vector2::new(0.0, 0.0),
            half_height: 1.0,
        }
    }
}

impl Camera2D {
    pub fn new(centre: Vector2<f32>, half_height: f32) -> Self {
        Self { centre, half_height }
    }

    /// World → view: translate world so the camera centre sits at the origin.
    pub fn view(&self) -> Matrix4<f32> {
        Matrix4::from_translation(Vector3::new(-self.centre.x, -self.centre.y, 0.0))
    }

    /// View → Vulkan clip space. `aspect = width / height` of the target.
    /// The depth range maps world z ∈ [-1, 1] → clip z ∈ [0, 1]; objects at
    /// z = 0 land mid-depth, which suits 2D scenes where everything sits on
    /// the z = 0 plane.
    pub fn proj(&self, aspect: f32) -> Matrix4<f32> {
        let half_width = self.half_height * aspect;
        Matrix4::new(
            1.0 / half_width, 0.0,                    0.0, 0.0,
            0.0,              1.0 / self.half_height, 0.0, 0.0,
            0.0,              0.0,                    0.5, 0.0,
            0.0,              0.0,                    0.5, 1.0,
        )
    }
}
