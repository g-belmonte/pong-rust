use std::any::Any;

use engine::scene::Behaviour;

/// Wall behaviour. Carries the wall's height so neighbouring behaviours
/// (paddles, ball) can read it via `scene.behaviour::<WallBehaviour>(id)`
/// without needing a separate bounds field on `Object`. Walls are static —
/// no `update` / `on_event` hooks.
pub struct WallBehaviour {
    pub height: f32,
}

impl WallBehaviour {
    pub fn new(height: f32) -> Self {
        Self { height }
    }
}

impl Behaviour for WallBehaviour {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
