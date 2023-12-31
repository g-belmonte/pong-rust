use cgmath::{Deg, Matrix4, Point3, Vector2, Vector3, Zero};
use num::clamp;
use rand::Rng;

use crate::ball::Ball;
use crate::camera::Camera;
use crate::graphics_manager::constants::{WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::graphics_manager::structures::ModelMesh;
use crate::paddle::Paddle;
use crate::wall::Wall;

pub struct ModelData {
    pub model_mesh: ModelMesh,
    pub model_transform: Matrix4<f32>,
}

pub struct Scene {
    pub camera: Camera,
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
    pub fn new() -> Self {
        Self {
            camera: Camera::new(
                Matrix4::look_at(
                    Point3::new(0.0, 0.0, 10.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0),
                ),
                cgmath::perspective(
                    Deg(45.0),
                    WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32,
                    0.1,
                    10.0,
                ),
            ),
            left_paddle: Paddle::new(
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
                Vector3 {
                    x: 0.0,
                    y: -3.2,
                    z: 0.0,
                },
                0.2,
                10.0,
            ),
            bottom_wall: Wall::new(
                Vector3 {
                    x: 0.0,
                    y: 3.2,
                    z: 0.0,
                },
                0.2,
                10.0,
            ),
            ball: Ball::new(Vector3::zero(), 0.2, color::GREEN),
        }
    }

    pub fn get_model_data(&self) -> Vec<ModelData> {
        vec![
            ModelData {
                model_mesh: self.left_paddle.model_mesh.clone(),
                model_transform: Matrix4::from_translation(self.left_paddle.position),
            },
            ModelData {
                model_mesh: self.right_paddle.model_mesh.clone(),
                model_transform: Matrix4::from_translation(self.right_paddle.position),
            },
            ModelData {
                model_mesh: self.top_wall.model_mesh.clone(),
                model_transform: Matrix4::from_translation(self.top_wall.position),
            },
            ModelData {
                model_mesh: self.bottom_wall.model_mesh.clone(),
                model_transform: Matrix4::from_translation(self.bottom_wall.position),
            },
            ModelData {
                model_mesh: self.ball.model_mesh.clone(),
                model_transform: Matrix4::from_translation(self.ball.position),
            },
        ]
    }

    pub fn get_model_transforms(&self) -> Vec<Matrix4<f32>> {
        vec![
            Matrix4::from_translation(self.left_paddle.position),
            Matrix4::from_translation(self.right_paddle.position),
            Matrix4::from_translation(self.top_wall.position),
            Matrix4::from_translation(self.bottom_wall.position),
            Matrix4::from_translation(self.ball.position),
        ]
    }

    pub fn update(&mut self, delta_time: f32) {
        // FIXME: this whole thing is buggy.
        // Sometimes the ball will penetrate a bit in the paddle and get stuck
        // Might even pierce its way through, given enough time

        // NOTE: positive Y is downwards, so upper_boundary < lower_boundary

        // geometry
        let upper_boundary = self.top_wall.position.y + (self.top_wall.height / 2.0);
        let lower_boundary = self.bottom_wall.position.y - (self.top_wall.height / 2.0);
        let hpw = self.left_paddle.width / 2.0; // Half paddle width. Both paddles have the same width.
        let hph = self.left_paddle.height / 2.0; // Half paddle height. Both paddles have the same height.
        let lpx = self.left_paddle.position.x; // Left paddle x position
        let lpy = self.left_paddle.position.y; // Left paddle y position
        let rpx = self.right_paddle.position.x; // Right paddle x position
        let rpy = self.right_paddle.position.y; // Right paddle y position
        let br = self.ball.side_length / 2.0; // Ball "radius"
        let bpx = self.ball.position.x; // Ball x position
        let bpy = self.ball.position.y; // Ball y position

        // CONTACT OF PADDLES AND WALLS
        // UPDATES PADDLE POSITION
        // Simulates paddles touching the walls by limiting the max and min values for the Y component of the paddle's position
        self.left_paddle.position.y = clamp(
            lpy + (delta_time * self.left_paddle.velocity),
            upper_boundary + hph,
            lower_boundary - hph,
        );
        self.right_paddle.position.y = clamp(
            self.right_paddle.position.y + (delta_time * self.right_paddle.velocity),
            upper_boundary + hph,
            lower_boundary - hph,
        );

        // CONTACT OF BALL AND OTHER OBJECTS
        // UPDATES BALL VELOCITY
        // Vertical contact - inverts Y component of velocity
        let is_touching_walls = self.ball.position.y + (self.ball.side_length / 2.0)
            > lower_boundary
            || self.ball.position.y - (self.ball.side_length / 2.0) < upper_boundary;

        let is_touching_top_of_left_paddle = false; // unimplemented
        let is_touching_top_of_right_paddle = false; // unimplemented

        let is_touching_bottom_of_left_paddle = false; // unimplemented
        let is_touching_bottom_of_right_paddle = false; // unimplemented

        let is_touching_bottom_or_top_of_paddles =
            is_touching_bottom_of_left_paddle ||
            is_touching_bottom_of_right_paddle ||
            is_touching_top_of_left_paddle ||
            is_touching_top_of_right_paddle;

        let is_vertical_contact =
            is_touching_walls || is_touching_bottom_or_top_of_paddles;
        if is_vertical_contact {
            self.ball.velocity.y *= -1.0;
        }

        // Horizontal contact - inverts X component of velocity
        let is_ball_touching_right_face_of_left_paddle =
            bpx - br < lpx + hpw && // ball touches the right face of the left paddle
            bpy + br > lpy - hph && // ball is under the top of the left paddle
            bpy - br < lpy + hph; // ball is over the bottom of the left paddle
        let is_ball_touching_left_face_of_right_paddle =
            bpx + br > rpx - hpw && // ball touches the left face of the right paddle
            bpy + br > rpy - hph && // ball is under the top of the right paddle
            bpy - br < rpy + hph;   // ball is over the bottom of the right paddle

        // NOTE: Ignore case where ball touches behind the paddle, per the game mechanics this would be impossible

        let is_horizontal_contact =
            is_ball_touching_right_face_of_left_paddle || is_ball_touching_left_face_of_right_paddle;
        if is_horizontal_contact {
            self.ball.velocity.x *= -1.0;
        }

        // Update position
        self.ball.position.x += delta_time * self.ball.velocity.x;
        self.ball.position.y += delta_time * self.ball.velocity.y;
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
