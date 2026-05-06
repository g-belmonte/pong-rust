use cgmath::Vector3;

use crate::graphics_manager::structures::{ModelMesh, Vertex};
use crate::graphics_manager::{GraphicsManager, ModelHandle};

pub struct Paddle {
    pub model_mesh: ModelMesh,
    pub model_handle: ModelHandle,
    pub position: Vector3<f32>,
    pub velocity: f32,
    pub height: f32,
    pub width: f32,
}

impl Paddle {
    pub fn new(
        gm: &mut GraphicsManager,
        position: Vector3<f32>,
        height: f32,
        width: f32,
        color: [f32; 3],
    ) -> Self {
        let model_mesh = ModelMesh {
            vertices: Paddle::vertices(height, width, color),
            indices: vec![0u32, 1, 2, 2, 3, 0],
        };
        let model_handle = gm.register_model(&model_mesh);
        Self {
            model_mesh,
            model_handle,
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
