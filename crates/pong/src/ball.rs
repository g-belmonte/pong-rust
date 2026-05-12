use std::any::Any;

use cgmath::Vector2;
use rand::Rng;

use engine::graphics_manager::GraphicsManager;
use engine::resources::{Resources, Texture};
use engine::scene::{Behaviour, ObjectId, Renderable, UpdateCtx};

use crate::paddle::PaddleBehaviour;
use crate::wall::WallBehaviour;

const BALL_TEXTURE_PNG: &[u8] = include_bytes!("../assets/tennis-ball.png");

pub struct BallBehaviour {
    pub velocity: Vector2<f32>,
    pub side_length: f32,
    // RAII: the texture must outlive every textured instance using it.
    // Stored here so the ball Object's lifetime governs the texture's.
    _texture: Texture,
    top_wall: ObjectId,
    bottom_wall: ObjectId,
    left_paddle: ObjectId,
    right_paddle: ObjectId,
    kickoff_speed_x: f32,
}

impl BallBehaviour {
    pub fn load(
        resources: &mut Resources,
        gm: &mut GraphicsManager,
        side_length: f32,
        kickoff_speed_x: f32,
        top_wall: ObjectId,
        bottom_wall: ObjectId,
        left_paddle: ObjectId,
        right_paddle: ObjectId,
    ) -> (Self, Renderable) {
        let texture = resources.load_texture_png(gm, BALL_TEXTURE_PNG);
        let renderable = Renderable::Textured {
            texture: texture.handle(),
            uv_offset: [0.0, 0.0],
            uv_scale: [1.0, 1.0],
        };
        (
            Self {
                velocity: Vector2 { x: 0.0, y: 0.0 },
                side_length,
                _texture: texture,
                top_wall,
                bottom_wall,
                left_paddle,
                right_paddle,
                kickoff_speed_x,
            },
            renderable,
        )
    }

    /// Randomised serve, called by [`PhaseController`] on Space at kickoff.
    pub fn kickoff(&mut self) {
        let mut rng = rand::thread_rng();
        let mut v = Vector2 {
            x: -self.kickoff_speed_x,
            y: rng.gen_range(-1.0..1.0),
        };
        if rand::random() {
            v.x = -v.x;
        }
        self.velocity = v;
    }

    pub fn stop(&mut self) {
        self.velocity = Vector2 { x: 0.0, y: 0.0 };
    }
}

impl Behaviour for BallBehaviour {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
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
                    .unwrap_or_else(|| cgmath::vec3(0.0, 0.0, 0.0));
                let (w, h) = ctx
                    .scene
                    .behaviour::<PaddleBehaviour>(pid)
                    .map(|p| (p.width, p.height))
                    .unwrap_or((0.0, 0.0));
                (pos.x, pos.y, w / 2.0, h / 2.0)
            });

        let (mut x, mut y) = match ctx.scene.get(ctx.self_id) {
            Some(obj) => (obj.transform.position.x, obj.transform.position.y),
            None => return,
        };
        let br = self.side_length / 2.0;

        // Substep so one fast tick cannot tunnel through a paddle. Step size
        // is bounded by the smaller of the two paddle widths.
        let smallest_half_w = paddles[0].2.min(paddles[1].2);
        let speed = (self.velocity.x.powi(2) + self.velocity.y.powi(2)).sqrt();
        let travel = speed * ctx.time;
        let n = if travel > 0.0 && smallest_half_w > 0.0 {
            ((travel / (smallest_half_w * 2.0)).ceil() as u32).max(1)
        } else {
            1
        };
        let sub_dt = ctx.time / n as f32;

        for _ in 0..n {
            let mut new_x = x + sub_dt * self.velocity.x;
            let mut new_y = y + sub_dt * self.velocity.y;

            // Walls (positive Y is downwards, so upper has smaller y).
            if new_y - br < upper {
                new_y = upper + br;
                if self.velocity.y < 0.0 {
                    self.velocity.y = -self.velocity.y;
                }
            } else if new_y + br > lower {
                new_y = lower - br;
                if self.velocity.y > 0.0 {
                    self.velocity.y = -self.velocity.y;
                }
            }

            // Paddles. Reflect on the axis with the smaller penetration depth.
            for &(px, py, hpw, hph) in &paddles {
                let overlap_x = (new_x + br).min(px + hpw) - (new_x - br).max(px - hpw);
                let overlap_y = (new_y + br).min(py + hph) - (new_y - br).max(py - hph);
                if overlap_x <= 0.0 || overlap_y <= 0.0 {
                    continue;
                }
                if overlap_x < overlap_y {
                    if new_x < px {
                        new_x -= overlap_x;
                        if self.velocity.x > 0.0 {
                            self.velocity.x = -self.velocity.x;
                        }
                    } else {
                        new_x += overlap_x;
                        if self.velocity.x < 0.0 {
                            self.velocity.x = -self.velocity.x;
                        }
                    }
                } else if new_y < py {
                    new_y -= overlap_y;
                    if self.velocity.y > 0.0 {
                        self.velocity.y = -self.velocity.y;
                    }
                } else {
                    new_y += overlap_y;
                    if self.velocity.y < 0.0 {
                        self.velocity.y = -self.velocity.y;
                    }
                }
                break;
            }

            x = new_x;
            y = new_y;
        }

        if let Some(obj) = ctx.scene.get_mut(ctx.self_id) {
            obj.transform.position.x = x;
            obj.transform.position.y = y;
        }
    }
}

