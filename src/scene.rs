//! Game scene: physics, state, and the registry of game objects.
//!
//! ## Coordinate convention
//!
//! **Positive Y is downwards.** This is load-bearing throughout the project:
//! - the top wall sits at `y = -WALL_OFFSET_Y`, the bottom at `y = +WALL_OFFSET_Y`,
//! - `Action::LeftPaddleUp` sets a *negative* y-velocity,
//! - the digit scoreboard sits *above* the play field at `y ≈ -3.7`.
//!
//! The convention matches the Vulkan clip-space Y direction the renderer uses,
//! so the scene's world coordinates and the framebuffer agree without flipping.
//! New code in this layer must follow the same convention; `digit.rs` and
//! `text.rs` mirror it.

use cgmath::{Matrix4, Vector2, Vector3, Zero};
use num::clamp;
use rand::Rng;

use crate::ball::Ball;
use crate::digit::{Digit, DigitMeshes};
use crate::graphics_manager::{GraphicsManager, ModelHandle};
use crate::paddle::Paddle;
use crate::text::{FontAtlas, TextLabel};
use crate::wall::Wall;

const PADDLE_HEIGHT: f32 = 2.0;
const PADDLE_WIDTH: f32 = 0.2;
const WALL_HEIGHT: f32 = 0.2;
const WALL_WIDTH: f32 = 10.0;
const DIGIT_SEGMENT_SIZE: f32 = 0.4;

// Walls are placed symmetrically around y = 0; positive Y is downwards, so
// the top wall is at -WALL_OFFSET_Y and the bottom at +WALL_OFFSET_Y.
const WALL_OFFSET_Y: f32 = 3.2;
// Ball's |x| past this counts as a goal — slightly outside paddle x (±4.0)
// so the ball visibly leaves the play field before scoring.
const GOAL_LINE_X: f32 = 4.7;
// Paddle vertical speed in world units / second when a movement key is held.
const PADDLE_SPEED: f32 = 2.0;
// Ball horizontal speed at kickoff. Sign is randomised; vertical component
// is sampled in (-1.0, 1.0).
const BALL_KICKOFF_SPEED_X: f32 = 4.0;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");
const FONT_RASTER_PX: f32 = 48.0;
// Pixels per world unit. 48 px / 0.05 = 960 px per world unit gives "Welcome"
// (~340 px wide) ~1.8 world units across — comfortably inside the play field.
const FONT_WORLD_SCALE: f32 = 0.005;

pub const WINNING_SCORE: u8 = 9;

pub struct Scene {
    pub left_paddle: Paddle,
    pub right_paddle: Paddle,
    pub top_wall: Wall,
    pub bottom_wall: Wall,
    pub ball: Ball,
    pub left_digit: Digit,
    pub right_digit: Digit,
    pub left_score: u8,
    pub right_score: u8,
    // Atlas owns the font texture; labels reference it via TextureHandle.
    // FUTURE (CLAUDE.md option-B follow-up): move GamePhase into Scene and
    // toggle label visibility internally on phase transitions.
    pub _font_atlas: FontAtlas,
    pub welcome_label: TextLabel,
    pub game_over_label: TextLabel,
}

mod color {
    pub const RED: [f32; 3] = [1.0, 0.0, 0.0];
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
    ResetRound,
    ResetGame,
}

impl Scene {
    pub fn new(gm: &mut GraphicsManager) -> Self {
        let font_atlas = FontAtlas::build(gm, FONT_BYTES, FONT_RASTER_PX);
        // Labels sit inside the play field, vertically centred.
        let welcome_label = TextLabel::new(
            gm,
            &font_atlas,
            "Welcome",
            Vector3 { x: 0.0, y: -1.2, z: 0.0 },
            FONT_WORLD_SCALE,
        );
        let mut game_over_label = TextLabel::new(
            gm,
            &font_atlas,
            "Game Over",
            Vector3 { x: 0.0, y: -1.2, z: 0.0 },
            FONT_WORLD_SCALE,
        );
        game_over_label.set_visible(false);

        // Register each shared solid-colour mesh exactly once. Both paddles
        // share one mesh, both walls share one, and all digit segments share
        // two (horizontal/vertical). Per-instance colour is supplied later.
        let paddle_mesh = Paddle::register_mesh(gm, PADDLE_HEIGHT, PADDLE_WIDTH);
        let wall_mesh = Wall::register_mesh(gm, WALL_HEIGHT, WALL_WIDTH);
        let digit_meshes = DigitMeshes::register(gm, DIGIT_SEGMENT_SIZE);

        Self {
            left_paddle: Paddle::with_mesh(
                gm,
                paddle_mesh,
                Vector3 { x: -4.0, y: 0.0, z: 0.0 },
                PADDLE_HEIGHT,
                PADDLE_WIDTH,
                color::RED,
            ),
            right_paddle: Paddle::with_mesh(
                gm,
                paddle_mesh,
                Vector3 { x: 4.0, y: 0.0, z: 0.0 },
                PADDLE_HEIGHT,
                PADDLE_WIDTH,
                color::BLUE,
            ),
            top_wall: Wall::with_mesh(
                gm,
                wall_mesh,
                Vector3 { x: 0.0, y: -WALL_OFFSET_Y, z: 0.0 },
                WALL_HEIGHT,
            ),
            bottom_wall: Wall::with_mesh(
                gm,
                wall_mesh,
                Vector3 { x: 0.0, y: WALL_OFFSET_Y, z: 0.0 },
                WALL_HEIGHT,
            ),
            ball: Ball::new(gm, Vector3::zero(), 0.2),
            left_digit: Digit::new(
                gm,
                &digit_meshes,
                Vector3 { x: -1.0, y: -3.7, z: 0.0 },
            ),
            right_digit: Digit::new(
                gm,
                &digit_meshes,
                Vector3 { x: 1.0, y: -3.7, z: 0.0 },
            ),
            left_score: 0,
            right_score: 0,
            _font_atlas: font_atlas,
            welcome_label,
            game_over_label,
        }
    }

    pub fn set_welcome_visible(&mut self, visible: bool) {
        self.welcome_label.set_visible(visible);
    }

    pub fn set_game_over_visible(&mut self, visible: bool) {
        self.game_over_label.set_visible(visible);
    }

    pub fn get_model_transforms(&self) -> Vec<(ModelHandle, Matrix4<f32>)> {
        let mut transforms = vec![
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
            (self.ball.model_handle, self.ball.model_matrix()),
        ];
        transforms.extend(self.left_digit.get_model_transforms());
        transforms.extend(self.right_digit.get_model_transforms());
        transforms.extend(self.welcome_label.get_model_transforms());
        transforms.extend(self.game_over_label.get_model_transforms());
        transforms
    }

    pub fn update(&mut self, delta_time: f32) {
        // upper_boundary < lower_boundary because positive Y is downwards
        // (see module-level doc).
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
        self.ball.position.x > GOAL_LINE_X || self.ball.position.x < -GOAL_LINE_X
    }

    pub fn match_over(&self) -> bool {
        self.left_score >= WINNING_SCORE || self.right_score >= WINNING_SCORE
    }

    fn reset_positions(&mut self) {
        self.ball.position = Vector3::zero();
        self.left_paddle.position.y = 0.0;
        self.right_paddle.position.y = 0.0;
    }

    pub fn handle_action(&mut self, action: Action) {
        match action {
            Action::LeftPaddleUp => self.left_paddle.velocity = -PADDLE_SPEED,
            Action::LeftPaddleDown => self.left_paddle.velocity = PADDLE_SPEED,
            Action::LeftPaddleStop => self.left_paddle.velocity = 0.0,
            Action::RightPaddleUp => self.right_paddle.velocity = -PADDLE_SPEED,
            Action::RightPaddleDown => self.right_paddle.velocity = PADDLE_SPEED,
            Action::RightPaddleStop => self.right_paddle.velocity = 0.0,
            Action::Kickoff => {
                let mut rng = rand::thread_rng();
                self.ball.velocity = Vector2 {
                    x: -BALL_KICKOFF_SPEED_X,
                    y: rng.gen_range(-1.0..1.0),
                };
                if rand::random() {
                    self.ball.velocity.x *= -1.0;
                }
            }
            Action::GameOver => {
                if self.ball.position.x > GOAL_LINE_X {
                    self.left_score += 1;
                    self.left_digit.set_value(self.left_score);
                } else if self.ball.position.x < -GOAL_LINE_X {
                    self.right_score += 1;
                    self.right_digit.set_value(self.right_score);
                }
                self.ball.velocity = cgmath::vec2(0.0, 0.0);
                self.left_paddle.velocity = 0.0;
                self.right_paddle.velocity = 0.0;
            }
            Action::ResetRound => self.reset_positions(),
            Action::ResetGame => {
                self.reset_positions();
                self.left_score = 0;
                self.right_score = 0;
                self.left_digit.set_value(0);
                self.right_digit.set_value(0);
            }
        }
    }
}
