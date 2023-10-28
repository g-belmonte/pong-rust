use cgmath::Vector3;

use crate::graphics_manager::structures::{ModelMesh, Vertex};

const INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];

pub struct Paddle {
    pub model_mesh: ModelMesh,
    pub position: Vector3<f32>,
    pub velocity: f32,
}

impl Paddle {
    pub fn new(position: Vector3<f32>, color: [f32; 3]) -> Self {
        Self {
            model_mesh: ModelMesh {
                vertices: Paddle::vertices(color),
                indices: INDICES_DATA,
            },
            position,
            velocity: 0.0,
        }
    }

    fn vertices(color: [f32; 3]) -> [Vertex; 4] {
        [
            Vertex {
                pos: [-0.1, -0.5],
                color,
            },
            Vertex {
                pos: [0.1, -0.5],
                color,
            },
            Vertex {
                pos: [0.1, 0.5],
                color,
            },
            Vertex {
                pos: [-0.1, 0.5],
                color,
            },
        ]
    }
}
