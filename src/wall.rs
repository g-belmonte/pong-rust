use cgmath::Vector3;

use crate::graphics_manager::structures::{ModelMesh, Vertex};

const INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];

pub struct Wall {
    pub model_mesh: ModelMesh,
    pub position: Vector3<f32>,
    pub height: f32,
    pub width: f32,
}

impl Wall {
    pub fn new(position: Vector3<f32>, height: f32, width: f32) -> Self {
        Self {
            model_mesh: ModelMesh {
                vertices: Wall::vertices(height, width),
                indices: INDICES_DATA,
            },
            position,
            height,
            width,
        }
    }

    fn vertices(height: f32, width: f32) -> [Vertex; 4] {
        let half_height = height / 2.0;
        let half_width = width / 2.0;

        [
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
