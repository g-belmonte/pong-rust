use cgmath::{Vector2, Vector3};

use crate::graphics_manager::structures::{ModelMesh, Vertex};

const INDICES_DATA: [u32; 6] = [0, 1, 2, 2, 3, 0];

pub struct Ball {
    pub model_mesh: ModelMesh,
    pub position: Vector3<f32>,
    pub velocity: Vector2<f32>,
    pub side_length: f32,
}

impl Ball {
    pub fn new(position: Vector3<f32>, side_length: f32, color: [f32; 3]) -> Self {
        Self {
            model_mesh: ModelMesh {
                vertices: Ball::vertices(side_length, color),
                indices: INDICES_DATA,
            },
            position,
            velocity: Vector2 { x: 0.0, y: 0.0 },
            side_length,
        }
    }

    fn vertices(side_length: f32, color: [f32; 3]) -> [Vertex; 4] {
        let radius = side_length / 2.0;

        [
            Vertex {
                pos: [-radius, -radius],
                color,
            },
            Vertex {
                pos: [radius, -radius],
                color,
            },
            Vertex {
                pos: [radius, radius],
                color,
            },
            Vertex {
                pos: [-radius, radius],
                color,
            },
        ]
    }
}
