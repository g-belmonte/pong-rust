use cgmath::Vector3;

use engine::graphics_manager::{GraphicsManager, MeshHandle, ModelHandle};

const COLOR: [f32; 3] = [0.0, 1.0, 0.0];

pub struct Wall {
    pub model_handle: ModelHandle,
    pub position: Vector3<f32>,
    pub height: f32,
}

impl Wall {
    pub fn with_mesh(
        gm: &mut GraphicsManager,
        mesh: MeshHandle,
        position: Vector3<f32>,
        height: f32,
    ) -> Self {
        let model_handle = gm.register_instance(mesh, COLOR);
        Self {
            model_handle,
            position,
            height,
        }
    }
}
