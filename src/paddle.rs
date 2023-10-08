use cgmath::Matrix4;

use crate::graphics_manager::structures::Vertex;

const VERTICES_DATA: [Vertex; 4] = [
    Vertex {
        pos: [-0.2, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.2, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [0.2, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        pos: [-0.2, 0.5],
        color: [1.0, 1.0, 1.0],
    },
];
const INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];

#[derive(Clone)]
pub struct ModelMesh {
    pub vertices: [Vertex; 4],
    pub indices: [u32; 6],
}

pub struct Paddle {
    pub model_mesh: ModelMesh,
    pub model_transform: Matrix4<f32>,
}

impl Paddle {
    pub fn new(model_transform: Matrix4<f32>) -> Self {
        Self {
            model_mesh: ModelMesh {
                vertices: VERTICES_DATA,
                indices: INDICES_DATA,
            },
            model_transform,
        }
    }
}
