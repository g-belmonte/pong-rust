use std::any::Any;

use num::clamp;

use engine::scene::{Behaviour, Event, KeyCode, ObjectId, UpdateCtx};

use crate::wall::WallBehaviour;

/// Paddle behaviour: reads its own `Object::transform`, integrates vertical
/// velocity, clamps against the top/bottom walls. Keys are configurable so
/// the same behaviour drives both players.
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

    fn on_event(&mut self, _ctx: &mut UpdateCtx, event: &Event) {
        match event {
            Event::KeyPressed(k) if *k == self.up_key => self.velocity = -self.speed,
            Event::KeyPressed(k) if *k == self.down_key => self.velocity = self.speed,
            // Either-key release stops the paddle (matches the original input
            // model: holding W or S sets a velocity, releasing either zeroes it).
            Event::KeyReleased(k) if *k == self.up_key || *k == self.down_key => {
                self.velocity = 0.0
            }
            _ => {}
        }
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        // Wall positions can shift only via deliberate scene edits (they
        // don't move during play), but reading them every frame keeps this
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
        // Positive Y is downwards, so the top boundary is the smaller y.
        let upper = top_y + top_h / 2.0;
        let lower = bottom_y - bottom_h / 2.0;
        let half_h = self.height / 2.0;

        if let Some(obj) = ctx.scene.get_mut(ctx.self_id) {
            let new_y = obj.transform.position.y + ctx.time * self.velocity;
            obj.transform.position.y = clamp(new_y, upper + half_h, lower - half_h);
        }
    }
}
