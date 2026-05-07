use cgmath::{Vector2, Vector3};

use crate::graphics_manager::structures::{TexturedModelMesh, TexturedVertex};
use crate::graphics_manager::{GraphicsManager, ModelHandle, TextureHandle};

const BALL_TEXTURE_PNG: &[u8] = include_bytes!("../assets/tennis-ball.png");

pub struct Ball {
    pub model_handle: ModelHandle,
    pub _texture: TextureHandle,
    pub position: Vector3<f32>,
    pub velocity: Vector2<f32>,
    pub side_length: f32,
}

impl Ball {
    pub fn new(
        gm: &mut GraphicsManager,
        position: Vector3<f32>,
        side_length: f32,
    ) -> Self {
        let texture = gm.register_texture(BALL_TEXTURE_PNG);
        let model_mesh = TexturedModelMesh {
            vertices: Ball::vertices(side_length),
            indices: vec![0u32, 1, 2, 2, 3, 0],
        };
        let model_handle = gm.register_textured_model_with(&model_mesh, texture);
        Self {
            model_handle,
            _texture: texture,
            position,
            velocity: Vector2 { x: 0.0, y: 0.0 },
            side_length,
        }
    }

    fn vertices(side_length: f32) -> Vec<TexturedVertex> {
        let radius = side_length / 2.0;

        vec![
            TexturedVertex {
                pos: [-radius, -radius],
                uv: [0.0, 0.0],
            },
            TexturedVertex {
                pos: [radius, -radius],
                uv: [1.0, 0.0],
            },
            TexturedVertex {
                pos: [radius, radius],
                uv: [1.0, 1.0],
            },
            TexturedVertex {
                pos: [-radius, radius],
                uv: [0.0, 1.0],
            },
        ]
    }
}
