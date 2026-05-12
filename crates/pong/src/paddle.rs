use std::any::Any;

use num::clamp;

use engine::input::KeyCode;
use engine::scene::{Behaviour, ObjectId, UpdateCtx};

use crate::wall::WallBehaviour;

/// Paddle behaviour: reads its own `Object::transform`, integrates vertical
/// velocity, clamps against the top/bottom walls. Keys are configurable so
/// the same behaviour drives both players.
///
/// Input is polled in `fixed_update`: holding the up/down key drives a
/// constant velocity, releasing both stops the paddle. The physics step
/// integrates with `ctx.time.delta_time()` (the constant fixed step).
pub struct PaddleBehaviour {
    pub velocity: f32,
    pub width: f32,
    pub height: f32,
    speed: f32,
    up_key: KeyCode,
    down_key: KeyCode,
    top_wall: ObjectId,
    bottom_wall: ObjectId,
}

impl PaddleBehaviour {
    pub fn new(
        width: f32,
        height: f32,
        speed: f32,
        up_key: KeyCode,
        down_key: KeyCode,
        top_wall: ObjectId,
        bottom_wall: ObjectId,
    ) -> Self {
        Self {
            velocity: 0.0,
            width,
            height,
            speed,
            up_key,
            down_key,
            top_wall,
            bottom_wall,
        }
    }

    pub fn set_velocity(&mut self, v: f32) {
        self.velocity = v;
    }
}

impl Behaviour for PaddleBehaviour {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fixed_update(&mut self, ctx: &mut UpdateCtx) {
        // Polling input: each fixed tick resolves the current key state.
        // Positive Y is downwards, so up-key produces a negative velocity.
        let up = ctx.input.is_pressed(self.up_key);
        let down = ctx.input.is_pressed(self.down_key);
        self.velocity = match (up, down) {
            (true, false) => -self.speed,
            (false, true) => self.speed,
            _ => 0.0,
        };

        // Wall positions can shift only via deliberate scene edits (they
        // don't move during play), but reading them every tick keeps this
        // honest if a future game animates them.
        let top_y = ctx.scene.get(self.top_wall).map(|o| o.transform.position.y).unwrap_or(0.0);
        let top_h = ctx
            .scene
            .behaviour::<WallBehaviour>(self.top_wall)
            .map(|w| w.height)
            .unwrap_or(0.0);
        let bottom_y = ctx
            .scene
            .get(self.bottom_wall)
            .map(|o| o.transform.position.y)
            .unwrap_or(0.0);
        let bottom_h = ctx
            .scene
            .behaviour::<WallBehaviour>(self.bottom_wall)
            .map(|w| w.height)
            .unwrap_or(0.0);
        let upper = top_y + top_h / 2.0;
        let lower = bottom_y - bottom_h / 2.0;
        let half_h = self.height / 2.0;
        let dt = ctx.time.delta_time();

        if let Some(obj) = ctx.scene.get_mut(ctx.self_id) {
            let new_y = obj.transform.position.y + dt * self.velocity;
            obj.transform.position.y = clamp(new_y, upper + half_h, lower - half_h);
        }
    }
}
