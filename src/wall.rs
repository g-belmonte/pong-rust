use cgmath::Vector3;

use crate::graphics_manager::structures::rect_mesh;
use crate::graphics_manager::{GraphicsManager, MeshHandle, ModelHandle};

const COLOR: [f32; 3] = [0.0, 1.0, 0.0];

pub struct Wall {
    pub model_handle: ModelHandle,
    pub position: Vector3<f32>,
    pub height: f32,
}

impl Wall {
    pub fn register_mesh(gm: &mut GraphicsManager, height: f32, width: f32) -> MeshHandle {
        gm.register_mesh(&rect_mesh(width, height))
    }

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
