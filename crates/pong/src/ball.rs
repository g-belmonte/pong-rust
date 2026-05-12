use cgmath::{Matrix4, Vector2, Vector3};

use engine::graphics_manager::{GraphicsManager, ModelHandle};
use engine::resources::{Resources, Texture};

const BALL_TEXTURE_PNG: &[u8] = include_bytes!("../assets/tennis-ball.png");

pub struct Ball {
    pub model_handle: ModelHandle,
    // RAII: dropped texture queues its GPU resources for cleanup.
    pub _texture: Texture,
    pub position: Vector3<f32>,
    pub velocity: Vector2<f32>,
    pub side_length: f32,
}

impl Ball {
    pub fn new(
        resources: &mut Resources,
        gm: &mut GraphicsManager,
        position: Vector3<f32>,
        side_length: f32,
    ) -> Self {
        let texture = resources.load_texture_png(gm, BALL_TEXTURE_PNG);
        // Whole texture maps onto the unit quad: uv goes 0..1 across both axes.
        let model_handle = gm.register_textured_instance(texture.handle(), [0.0, 0.0], [1.0, 1.0]);
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
