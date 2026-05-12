//! Game-state controller behaviour.
//!
//! Owns `GamePhase`, the score state, and the shared RAII resources (mesh /
//! atlas wrappers) for the scene. Drives the cross-object choreography that
//! used to live in `main.rs`: showing/hiding labels on phase transitions,
//! kicking off the ball, scoring, resetting positions, ending the match.

use std::any::Any;

use cgmath::Vector3;

use engine::input::KeyCode;
use engine::resources::Mesh;
use engine::scene::{Behaviour, ObjectId, UpdateCtx};

use crate::ball::BallBehaviour;
use crate::digit::{DigitBehaviour, DigitMeshes};
use crate::paddle::PaddleBehaviour;
use crate::text::{FontAtlas, TextLabelBehaviour};

#[derive(PartialEq)]
pub enum GamePhase {
    Start,
    Playing,
    End,
}

pub struct PhaseController {
    phase: GamePhase,
    left_score: u8,
    right_score: u8,
    winning_score: u8,
    goal_line_x: f32,

    // Object IDs captured at scene-build time.
    ball: ObjectId,
    left_paddle: ObjectId,
    right_paddle: ObjectId,
    left_digit: ObjectId,
    right_digit: ObjectId,
    welcome_label: ObjectId,
    game_over_label: ObjectId,

    // RAII resources owned by the controller so they outlive every instance
    // built against them. Underscore-prefixed: they're referenced by handle()
    // through the registered instances, not used directly here.
    _paddle_mesh: Mesh,
    _wall_mesh: Mesh,
    _digit_meshes: DigitMeshes,
    _font_atlas: FontAtlas,
}

#[allow(clippy::too_many_arguments)]
impl PhaseController {
    pub fn new(
        ball: ObjectId,
        left_paddle: ObjectId,
        right_paddle: ObjectId,
        left_digit: ObjectId,
        right_digit: ObjectId,
        welcome_label: ObjectId,
        game_over_label: ObjectId,
        winning_score: u8,
        goal_line_x: f32,
        paddle_mesh: Mesh,
        wall_mesh: Mesh,
        digit_meshes: DigitMeshes,
        font_atlas: FontAtlas,
    ) -> Self {
        Self {
            phase: GamePhase::Start,
            left_score: 0,
            right_score: 0,
            winning_score,
            goal_line_x,
            ball,
            left_paddle,
            right_paddle,
            left_digit,
            right_digit,
            welcome_label,
            game_over_label,
            _paddle_mesh: paddle_mesh,
            _wall_mesh: wall_mesh,
            _digit_meshes: digit_meshes,
            _font_atlas: font_atlas,
        }
    }

    fn set_label_visible(ctx: &mut UpdateCtx, id: ObjectId, visible: bool) {
        if let Some(label) = ctx.scene.behaviour_mut::<TextLabelBehaviour>(id) {
            label.set_visible(visible);
        }
    }

    fn reset_positions(ctx: &mut UpdateCtx, ball: ObjectId, paddles: [ObjectId; 2]) {
        if let Some(obj) = ctx.scene.get_mut(ball) {
            obj.transform.position = Vector3 { x: 0.0, y: 0.0, z: 0.0 };
        }
        if let Some(ball_b) = ctx.scene.behaviour_mut::<BallBehaviour>(ball) {
            ball_b.stop();
        }
        for pid in paddles {
            if let Some(obj) = ctx.scene.get_mut(pid) {
                obj.transform.position.y = 0.0;
            }
            if let Some(p) = ctx.scene.behaviour_mut::<PaddleBehaviour>(pid) {
                p.set_velocity(0.0);
            }
        }
    }

    fn ball_x(ctx: &UpdateCtx, ball: ObjectId) -> f32 {
        ctx.scene.get(ball).map(|o| o.transform.position.x).unwrap_or(0.0)
    }

    fn match_over(&self) -> bool {
        self.left_score >= self.winning_score || self.right_score >= self.winning_score
    }
}

impl Behaviour for PhaseController {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        // Edge inputs: Escape exits, Space drives phase transitions. Polled
        // from `update` (not `fixed_update`) so a tapped Space is consumed
        // once per frame, not once per fixed substep.
        if ctx.input.was_just_pressed(KeyCode::Escape) {
            ctx.request_exit();
            return;
        }
        if ctx.input.was_just_pressed(KeyCode::Space) {
            match self.phase {
                GamePhase::Start => {
                    self.phase = GamePhase::Playing;
                    Self::set_label_visible(ctx, self.welcome_label, false);
                    if let Some(ball) = ctx.scene.behaviour_mut::<BallBehaviour>(self.ball) {
                        ball.kickoff();
                    }
                }
                GamePhase::Playing => {}
                GamePhase::End => {
                    self.phase = GamePhase::Start;
                    Self::set_label_visible(ctx, self.game_over_label, false);
                    Self::set_label_visible(ctx, self.welcome_label, true);
                    self.left_score = 0;
                    self.right_score = 0;
                    if let Some(d) = ctx.scene.behaviour_mut::<DigitBehaviour>(self.left_digit) {
                        d.set_value(0);
                    }
                    if let Some(d) = ctx.scene.behaviour_mut::<DigitBehaviour>(self.right_digit) {
                        d.set_value(0);
                    }
                    Self::reset_positions(
                        ctx,
                        self.ball,
                        [self.left_paddle, self.right_paddle],
                    );
                }
            }
        }

        if self.phase != GamePhase::Playing {
            return;
        }
        let bx = Self::ball_x(ctx, self.ball);
        if bx <= self.goal_line_x && bx >= -self.goal_line_x {
            return;
        }
        // Score for whichever side the ball just exited past.
        if bx > self.goal_line_x {
            self.left_score += 1;
            if let Some(d) = ctx.scene.behaviour_mut::<DigitBehaviour>(self.left_digit) {
                d.set_value(self.left_score);
            }
        } else {
            self.right_score += 1;
            if let Some(d) = ctx.scene.behaviour_mut::<DigitBehaviour>(self.right_digit) {
                d.set_value(self.right_score);
            }
        }
        // Stop the ball + paddles regardless of outcome.
        if let Some(ball) = ctx.scene.behaviour_mut::<BallBehaviour>(self.ball) {
            ball.stop();
        }
        for pid in [self.left_paddle, self.right_paddle] {
            if let Some(p) = ctx.scene.behaviour_mut::<PaddleBehaviour>(pid) {
                p.set_velocity(0.0);
            }
        }

        if self.match_over() {
            self.phase = GamePhase::End;
            Self::set_label_visible(ctx, self.game_over_label, true);
        } else {
            self.phase = GamePhase::Start;
            Self::reset_positions(
                ctx,
                self.ball,
                [self.left_paddle, self.right_paddle],
            );
            Self::set_label_visible(ctx, self.welcome_label, true);
        }
    }
}
