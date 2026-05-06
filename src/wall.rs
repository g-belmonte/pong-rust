use cgmath::Vector3;

use crate::graphics_manager::structures::{ModelMesh, Vertex};
use crate::graphics_manager::{GraphicsManager, ModelHandle};

pub struct Wall {
    pub model_handle: ModelHandle,
    pub position: Vector3<f32>,
    pub height: f32,
}

impl Wall {
    pub fn new(
        gm: &mut GraphicsManager,
        position: Vector3<f32>,
        height: f32,
        width: f32,
    ) -> Self {
        let model_mesh = ModelMesh {
            vertices: Wall::vertices(height, width),
            indices: vec![0u32, 1, 2, 2, 3, 0],
        };
        let model_handle = gm.register_model(&model_mesh);
        Self {
            model_handle,
            position,
            height,
        }
    }

    fn vertices(height: f32, width: f32) -> Vec<Vertex> {
        let half_height = height / 2.0;
        let half_width = width / 2.0;

        vec![
            Vertex {
                pos: [-half_width, -half_height],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                pos: [half_width, -half_height],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                pos: [half_width, half_height],
                color: [0.0, 0.0, 1.0],
            },
            Vertex {
                pos: [-half_width, half_height],
                color: [1.0, 1.0, 1.0],
            },
        ]
    }
}
