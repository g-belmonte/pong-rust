use std::any::Any;

use engine::Vec2;
use rand::Rng;

use engine::audio::Sound;
use engine::graphics_manager::GraphicsManager;
use engine::resources::{Resources, Texture};
use engine::scene::{Behaviour, ObjectId, Renderable, UpdateCtx};

use crate::paddle::PaddleBehaviour;
use crate::wall::WallBehaviour;

// Max outgoing angle off the horizontal on a side hit, when the ball strikes
// the paddle's extreme edge. ~60°.
const PADDLE_REFLECT_MAX_ANGLE_RAD: f32 = std::f32::consts::FRAC_PI_3;
// Multiplicative speed bump applied on every paddle contact.
const SPEED_BOOST_PER_HIT: f32 = 1.05;
// Hard cap on |velocity|. With the fixed step of ~8.3 ms and paddle width 0.2,
// the no-substep invariant in `fixed_update` requires per-axis |v| < ~24.
// 16 leaves headroom even for fully diagonal trajectories.
const MAX_SPEED: f32 = 16.0;

pub struct BallBehaviour {
    pub velocity: Vec2,
    pub side_length: f32,
    // RAII: the texture must outlive every textured instance using it.
    // Stored here so the ball Object's lifetime governs the texture's.
    _texture: Texture,
    top_wall: ObjectId,
    bottom_wall: ObjectId,
    left_paddle: ObjectId,
    right_paddle: ObjectId,
    kickoff_speed_x: f32,
    wall_bounce_sfx: Sound,
    paddle_bounce_sfx: Sound,
}

impl BallBehaviour {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        resources: &mut Resources,
        gm: &mut GraphicsManager,
        side_length: f32,
        kickoff_speed_x: f32,
        top_wall: ObjectId,
        bottom_wall: ObjectId,
        left_paddle: ObjectId,
        right_paddle: ObjectId,
        wall_bounce_sfx: Sound,
        paddle_bounce_sfx: Sound,
    ) -> (Self, Renderable) {
        let texture = resources.load_texture_png(gm, engine::asset!("assets/tennis-ball.png"));
        let renderable = Renderable::Textured {
            texture: texture.handle(),
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
        };
        (
            Self {
                velocity: Vec2::ZERO,
                side_length,
                _texture: texture,
                top_wall,
                bottom_wall,
                left_paddle,
                right_paddle,
                kickoff_speed_x,
                wall_bounce_sfx,
                paddle_bounce_sfx,
            },
            renderable,
        )
    }

    /// Randomised serve, called by [`PhaseController`] on Space at kickoff.
    pub fn kickoff(&mut self) {
        let mut rng = rand::thread_rng();
        let mut v = Vec2::new(-self.kickoff_speed_x, rng.gen_range(-1.0..1.0));
        if rand::random() {
            v.x = -v.x;
        }
        self.velocity = v;
    }

    pub fn stop(&mut self) {
        self.velocity = Vec2::ZERO;
    }
}

impl Behaviour for BallBehaviour {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fixed_update(&mut self, ctx: &mut UpdateCtx) {
        // Snapshot the scene state the ball reacts to. Reading is cheap and
        // avoids a re-borrow against `ctx.scene` after we take &mut for self.
        let top_y = ctx
            .scene
            .get(self.top_wall)
            .map(|o| o.transform.position.y)
            .unwrap_or(0.0);
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

        let paddles: [(f32, f32, f32, f32); 2] = [self.left_paddle, self.right_paddle]
            .map(|pid| {
                let pos = ctx
                    .scene
                    .get(pid)
                    .map(|o| o.transform.position)
                    .unwrap_or(engine::Vec3::ZERO);
                let (w, h) = ctx
                    .scene
                    .behaviour::<PaddleBehaviour>(pid)
                    .map(|p| (p.width, p.height))
                    .unwrap_or((0.0, 0.0));
                (pos.x, pos.y, w / 2.0, h / 2.0)
            });

        let (x, y) = match ctx.scene.get(ctx.self_id) {
            Some(obj) => (obj.transform.position.x, obj.transform.position.y),
            None => return,
        };
        let br = self.side_length / 2.0;
        let dt = ctx.time.delta_time();

        // No substep: the fixed step (~8.3 ms at 120 Hz) is small enough that
        // `dt * |v|` cannot exceed the paddle width at any speed Pong uses.
        // The Phase 2 substep loop is gone — that's the headline win of
        // Phase 3.
        let mut new_x = x + dt * self.velocity.x;
        let mut new_y = y + dt * self.velocity.y;

        // Walls (positive Y is downwards, so upper has smaller y).
        let mut wall_hit = false;
        if new_y - br < upper {
            new_y = upper + br;
            if self.velocity.y < 0.0 {
                self.velocity.y = -self.velocity.y;
                wall_hit = true;
            }
        } else if new_y + br > lower {
            new_y = lower - br;
            if self.velocity.y > 0.0 {
                self.velocity.y = -self.velocity.y;
                wall_hit = true;
            }
        }
        if wall_hit {
            ctx.audio.play(&self.wall_bounce_sfx);
        }

        // Paddles. Reflect on the axis with the smaller penetration depth.
        let mut paddle_hit = false;
        for &(px, py, hpw, hph) in &paddles {
            let overlap_x = (new_x + br).min(px + hpw) - (new_x - br).max(px - hpw);
            let overlap_y = (new_y + br).min(py + hph) - (new_y - br).max(py - hph);
            if overlap_x <= 0.0 || overlap_y <= 0.0 {
                continue;
            }
            paddle_hit = true;
            if overlap_x < overlap_y {
                let push_left = new_x < px;
                if push_left {
                    new_x -= overlap_x;
                } else {
                    new_x += overlap_x;
                }
                let heading_in = if push_left {
                    self.velocity.x > 0.0
                } else {
                    self.velocity.x < 0.0
                };
                if heading_in {
                    // Position-based reflection: where the ball hits on the
                    // paddle face decides the outgoing angle.
                    let t = ((new_y - py) / hph).clamp(-1.0, 1.0);
                    let angle = t * PADDLE_REFLECT_MAX_ANGLE_RAD;
                    let speed = (self.velocity.length() * SPEED_BOOST_PER_HIT).min(MAX_SPEED);
                    let dir_x = if push_left { -1.0 } else { 1.0 };
                    self.velocity.x = dir_x * speed * angle.cos();
                    self.velocity.y = speed * angle.sin();
                }
            } else {
                // Top/bottom graze: simple reflect, same speed bump.
                let heading_in = if new_y < py {
                    new_y -= overlap_y;
                    self.velocity.y > 0.0
                } else {
                    new_y += overlap_y;
                    self.velocity.y < 0.0
                };
                if heading_in {
                    self.velocity.y = -self.velocity.y;
                    let speed = (self.velocity.length() * SPEED_BOOST_PER_HIT).min(MAX_SPEED);
                    if let Some(dir) = self.velocity.try_normalize() {
                        self.velocity = dir * speed;
                    }
                }
            }
            break;
        }
        if paddle_hit {
            ctx.audio.play(&self.paddle_bounce_sfx);
        }

        if let Some(obj) = ctx.scene.get_mut(ctx.self_id) {
            obj.transform.position.x = new_x;
            obj.transform.position.y = new_y;
        }
    }
}
