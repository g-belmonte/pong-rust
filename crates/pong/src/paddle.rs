use cgmath::Vector3;

use engine::graphics_manager::{GraphicsManager, MeshHandle, ModelHandle};

pub struct Paddle {
    pub model_handle: ModelHandle,
    pub position: Vector3<f32>,
    pub velocity: f32,
    pub height: f32,
    pub width: f32,
}

impl Paddle {
    /// Build a paddle that reuses a previously-loaded mesh.
    pub fn with_mesh(
        gm: &mut GraphicsManager,
        mesh: MeshHandle,
        position: Vector3<f32>,
        height: f32,
        width: f32,
        color: [f32; 3],
    ) -> Self {
        let model_handle = gm.register_instance(mesh, color);
        Self {
            model_handle,
            position,
            velocity: 0.0,
            height,
            width,
        }
    }
}
