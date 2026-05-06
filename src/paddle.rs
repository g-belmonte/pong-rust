use cgmath::Vector3;

use crate::graphics_manager::structures::{ModelMesh, Vertex};

pub struct Paddle {
    pub model_mesh: ModelMesh,
    pub position: Vector3<f32>,
    pub velocity: f32,
    pub height: f32,
    pub width: f32,
}

impl Paddle {
    pub fn new(position: Vector3<f32>, height: f32, width: f32, color: [f32; 3]) -> Self {
        Self {
            model_mesh: ModelMesh {
                vertices: Paddle::vertices(height, width, color),
                indices: vec![0u32, 1, 2, 2, 3, 0],
            },
            position,
            velocity: 0.0,
            height,
            width,
        }
    }

    fn vertices(height: f32, width: f32, color: [f32; 3]) -> Vec<Vertex> {
        let half_height = height / 2.0;
        let half_width = width / 2.0;

        vec![
            Vertex {
                pos: [-half_width, -half_height],
                color,
            },
            Vertex {
                pos: [half_width, -half_height],
                color,
            },
            Vertex {
                pos: [half_width, half_height],
                color,
            },
            Vertex {
                pos: [-half_width, half_height],
                color,
            },
        ]
    }
}
