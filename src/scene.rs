use cgmath::{Deg, Matrix4, Point3, SquareMatrix, Vector3, Zero};
use num::clamp;

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



pub enum Action {
    LeftPaddleUp,
    LeftPaddleDown,
    LeftPaddleStop,
    RightPaddleUp,
    RightPaddleDown,
    RightPaddleStop,
    Kickoff,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            camera: Camera::new(
                Matrix4::look_at(
                    Point3::new(0.0, 0.0, 4.0),
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
            left_paddle: Paddle::new(Vector3 {
                x: -3.7,
                y: 0.0,
                z: 0.0,
            }),
            right_paddle: Paddle::new(Vector3 {
                x: 3.7,
                y: 0.0,
                z: 0.0,
            }),
            top_wall: Wall::new(Vector3 {
                x: 0.0,
                y: -3.2,
                z: 0.0,
            }),
            bottom_wall: Wall::new(Vector3 {
                x: 0.0,
                y: 3.2,
                z: 0.0,
            }),
            ball: Ball::new(Vector3::zero()),
        }
    }

    pub fn get_model_data(&self) -> Vec<ModelData> {
        vec![
            ModelData {
                model_mesh: self.left_paddle.model_mesh.clone(),
                model_transform: Matrix4::<f32>::identity()
                    + Matrix4::from_translation(self.left_paddle.position),
            },
            ModelData {
                model_mesh: self.right_paddle.model_mesh.clone(),
                model_transform: Matrix4::<f32>::identity()
                    + Matrix4::from_translation(self.right_paddle.position),
            },
            ModelData {
                model_mesh: self.top_wall.model_mesh.clone(),
                model_transform: Matrix4::<f32>::identity()
                    + Matrix4::from_translation(self.top_wall.position),
            },
            ModelData {
                model_mesh: self.bottom_wall.model_mesh.clone(),
                model_transform: Matrix4::<f32>::identity()
                    + Matrix4::from_translation(self.bottom_wall.position),
            },
            ModelData {
                model_mesh: self.ball.model_mesh.clone(),
                model_transform: Matrix4::<f32>::identity()
                    + Matrix4::from_translation(self.ball.position),
            }
        ]
    }

    pub fn get_model_transforms(&self) -> Vec<Matrix4<f32>> {
        vec![
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.left_paddle.position),
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.right_paddle.position),
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.top_wall.position),
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.bottom_wall.position),
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.ball.position),
        ]
    }

    pub fn update(&mut self, delta_time: f32) {
        self.left_paddle.position.y = clamp(
            self.left_paddle.position.y + (delta_time * self.left_paddle.velocity),
            -2.0,
            2.0
        );
        self.right_paddle.position.y = clamp(
            self.right_paddle.position.y + (delta_time * self.right_paddle.velocity),
            -2.0,
            2.0
        );
        self.ball.position.x += delta_time * self.ball.velocity.x;
        self.ball.position.y += delta_time * self.ball.velocity.y;
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
            Action::Kickoff => self.ball.velocity = cgmath::vec2(1.0, 0.3),
        }
    }
}
