use cgmath::{Vector2, Vector3};

use crate::graphics_manager::structures::{ModelMesh, Vertex};
use crate::graphics_manager::{GraphicsManager, ModelHandle};

pub struct Ball {
    pub model_mesh: ModelMesh,
    pub model_handle: ModelHandle,
    pub position: Vector3<f32>,
    pub velocity: Vector2<f32>,
    pub side_length: f32,
}

impl Ball {
    pub fn new(
        gm: &mut GraphicsManager,
        position: Vector3<f32>,
        side_length: f32,
        color: [f32; 3],
    ) -> Self {
        let model_mesh = ModelMesh {
            vertices: Ball::vertices(side_length, color),
            indices: vec![0u32, 1, 2, 2, 3, 0],
        };
        let model_handle = gm.register_model(&model_mesh);
        Self {
            model_mesh,
            model_handle,
            position,
            velocity: Vector2 { x: 0.0, y: 0.0 },
            side_length,
        }
    }

    fn vertices(side_length: f32, color: [f32; 3]) -> Vec<Vertex> {
        let radius = side_length / 2.0;

        vec![
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
