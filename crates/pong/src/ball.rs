use cgmath::{Matrix4, Vector2, Vector3};

use engine::graphics_manager::{GraphicsManager, ModelHandle, TextureHandle};

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
        // Whole texture maps onto the unit quad: uv goes 0..1 across both axes.
        let model_handle = gm.register_textured_instance(texture, [0.0, 0.0], [1.0, 1.0]);
        Self {
            model_handle,
            _texture: texture,
            position,
            velocity: Vector2 { x: 0.0, y: 0.0 },
            side_length,
        }
    }

    /// Per-frame model matrix. The unit quad lives in `[-0.5..0.5]^2`, so
    /// scaling by `side_length` brings it to the ball's actual size; translation
    /// places it in the world.
    pub fn model_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_translation(self.position)
            * Matrix4::from_nonuniform_scale(self.side_length, self.side_length, 1.0)
    }
}
