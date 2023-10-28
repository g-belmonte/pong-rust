use cgmath::Vector3;

use crate::graphics_manager::structures::{ModelMesh, Vertex};

const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-0.1, -0.1],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.1, -0.1],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [0.1, 0.1],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.1, 0.1],
        color: [1.0, 1.0, 1.0],
    },
];
const INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];

pub struct Ball {
    pub model_mesh: ModelMesh,
    pub position: Vector3<f32>,
    pub velocity: f32,
}

impl Ball {
    pub fn new(position: Vector3<f32>) -> Self {
        Self {
            model_mesh: ModelMesh {
                vertices: VERTICES_DATA,
                indices: INDICES_DATA,
            },
            position,
            velocity: 0.0,
        }
    }
}
