use cgmath::{Matrix4, Vector2, Vector3, Zero};
use num::clamp;
use rand::Rng;

use crate::ball::Ball;
use crate::graphics_manager::{GraphicsManager, ModelHandle};
use crate::paddle::Paddle;
use crate::wall::Wall;

pub struct Scene {
    pub left_paddle: Paddle,
    pub right_paddle: Paddle,
    pub top_wall: Wall,
    pub bottom_wall: Wall,
    pub ball: Ball,
}

mod color {
    pub const RED: [f32; 3] = [1.0, 0.0, 0.0];
    pub const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
    pub const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
}

pub enum Action {
    LeftPaddleUp,
    LeftPaddleDown,
    LeftPaddleStop,
    RightPaddleUp,
    RightPaddleDown,
    RightPaddleStop,
    Kickoff,
    GameOver,
    ResetGame,
}

impl Scene {
    pub fn new(gm: &mut GraphicsManager) -> Self {
        Self {
            left_paddle: Paddle::new(
                gm,
                Vector3 {
                    x: -4.0,
                    y: 0.0,
                    z: 0.0,
                },
                2.0,
                0.2,
                color::RED,
            ),
            right_paddle: Paddle::new(
                gm,
                Vector3 {
                    x: 4.0,
                    y: 0.0,
                    z: 0.0,
                },
                2.0,
                0.2,
                color::BLUE,
            ),
            top_wall: Wall::new(
                gm,
                Vector3 {
                    x: 0.0,
                    y: -3.2,
                    z: 0.0,
                },
                0.2,
                10.0,
            ),
            bottom_wall: Wall::new(
                gm,
                Vector3 {
                    x: 0.0,
                    y: 3.2,
                    z: 0.0,
                },
                0.2,
                10.0,
            ),
            ball: Ball::new(gm, Vector3::zero(), 0.2, color::GREEN),
        }
    }

    pub fn get_model_transforms(&self) -> Vec<(ModelHandle, Matrix4<f32>)> {
        vec![
            (
                self.left_paddle.model_handle,
                Matrix4::from_translation(self.left_paddle.position),
            ),
            (
                self.right_paddle.model_handle,
                Matrix4::from_translation(self.right_paddle.position),
            ),
            (
                self.top_wall.model_handle,
                Matrix4::from_translation(self.top_wall.position),
            ),
            (
                self.bottom_wall.model_handle,
                Matrix4::from_translation(self.bottom_wall.position),
            ),
            (
                self.ball.model_handle,
                Matrix4::from_translation(self.ball.position),
            ),
        ]
    }

    pub fn update(&mut self, delta_time: f32) {
        // NOTE: positive Y is downwards, so upper_boundary < lower_boundary
        let upper_boundary = self.top_wall.position.y + (self.top_wall.height / 2.0);
        let lower_boundary = self.bottom_wall.position.y - (self.bottom_wall.height / 2.0);
        let hpw = self.left_paddle.width / 2.0;
        let hph = self.left_paddle.height / 2.0;
        let br = self.ball.side_length / 2.0;

        // Paddle position is clamped against the inner wall surfaces.
        self.left_paddle.position.y = clamp(
            self.left_paddle.position.y + delta_time * self.left_paddle.velocity,
            upper_boundary + hph,
            lower_boundary - hph,
        );
        self.right_paddle.position.y = clamp(
            self.right_paddle.position.y + delta_time * self.right_paddle.velocity,
            upper_boundary + hph,
            lower_boundary - hph,
        );

        // Substep so one fast tick cannot carry the ball clear through a paddle.
        let speed =
            (self.ball.velocity.x.powi(2) + self.ball.velocity.y.powi(2)).sqrt();
        let travel = speed * delta_time;
        let n = if travel > 0.0 {
            ((travel / self.left_paddle.width).ceil() as u32).max(1)
        } else {
            1
        };
        let sub_dt = delta_time / n as f32;

        for _ in 0..n {
            let mut new_x = self.ball.position.x + sub_dt * self.ball.velocity.x;
            let mut new_y = self.ball.position.y + sub_dt * self.ball.velocity.y;

            // Walls.
            if new_y - br < upper_boundary {
                new_y = upper_boundary + br;
                if self.ball.velocity.y < 0.0 {
                    self.ball.velocity.y = -self.ball.velocity.y;
                }
            } else if new_y + br > lower_boundary {
                new_y = lower_boundary - br;
                if self.ball.velocity.y > 0.0 {
                    self.ball.velocity.y = -self.ball.velocity.y;
                }
            }

            // Paddles. Reflect on the axis with the smaller penetration depth —
            // that's the face the ball just crossed. Only flip velocity when it
            // points into the paddle, otherwise repeated frames would buzz.
            for paddle in [&self.left_paddle, &self.right_paddle] {
                let px = paddle.position.x;
                let py = paddle.position.y;
                let overlap_x =
                    (new_x + br).min(px + hpw) - (new_x - br).max(px - hpw);
                let overlap_y =
                    (new_y + br).min(py + hph) - (new_y - br).max(py - hph);
                if overlap_x <= 0.0 || overlap_y <= 0.0 {
                    continue;
                }
                if overlap_x < overlap_y {
                    if new_x < px {
                        new_x -= overlap_x;
                        if self.ball.velocity.x > 0.0 {
                            self.ball.velocity.x = -self.ball.velocity.x;
                        }
                    } else {
                        new_x += overlap_x;
                        if self.ball.velocity.x < 0.0 {
                            self.ball.velocity.x = -self.ball.velocity.x;
                        }
                    }
                } else if new_y < py {
                    new_y -= overlap_y;
                    if self.ball.velocity.y > 0.0 {
                        self.ball.velocity.y = -self.ball.velocity.y;
                    }
                } else {
                    new_y += overlap_y;
                    if self.ball.velocity.y < 0.0 {
                        self.ball.velocity.y = -self.ball.velocity.y;
                    }
                }
                break;
            }

            self.ball.position.x = new_x;
            self.ball.position.y = new_y;
        }
    }

    pub fn game_over(&self) -> bool {
        self.ball.position.x > 4.7 || self.ball.position.x < -4.7
    }

    pub fn handle_action(&mut self, action: Action) {
        match action {
            // positive y is downwards
            Action::LeftPaddleUp => self.left_paddle.velocity = -2.0,
            Action::LeftPaddleDown => self.left_paddle.velocity = 2.0,
            Action::LeftPaddleStop => self.left_paddle.velocity = 0.0,
            Action::RightPaddleUp => self.right_paddle.velocity = -2.0,
            Action::RightPaddleDown => self.right_paddle.velocity = 2.0,
            Action::RightPaddleStop => self.right_paddle.velocity = 0.0,
            Action::Kickoff => {
                let mut rng = rand::thread_rng();
                self.ball.velocity = Vector2 {
                    x: -4.0,
                    y: rng.gen_range(-1.0..1.0),
                };
                if rand::random() {
                    self.ball.velocity.x *= -1.0;
                }
            }
            Action::GameOver => {
                self.ball.velocity = cgmath::vec2(0.0, 0.0);
                self.left_paddle.velocity = 0.0;
                self.right_paddle.velocity = 0.0;
            }
            Action::ResetGame => {
                self.ball.position.x = 0.0;
                self.ball.position.y = 0.0;
                self.left_paddle.position.y = 0.0;
                self.right_paddle.position.y = 0.0;
            }
        }
    }
}
