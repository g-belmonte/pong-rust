use cgmath::{Vector3, Vector2};

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
    pub velocity: Vector2<f32>,
}

impl Ball {
    pub fn new(position: Vector3<f32>) -> Self {
        Self {
            model_mesh: ModelMesh {
                vertices: VERTICES_DATA,
                indices: INDICES_DATA,
            },
            position,
            velocity: Vector2 { x: 0.0, y: 0.0  },
        }
    }
}
