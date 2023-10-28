use cgmath::{Deg, Matrix4, Point3, SquareMatrix, Vector3};

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
}



pub enum Action {
    LeftPaddleUp,
    LeftPaddleDown,
    LeftPaddleStop,
    RightPaddleUp,
    RightPaddleDown,
    RightPaddleStop,
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
            }
        ]
    }

    pub fn get_model_transforms(&self) -> Vec<Matrix4<f32>> {
        vec![
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.left_paddle.position),
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.right_paddle.position),
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.top_wall.position),
            Matrix4::<f32>::identity() + Matrix4::from_translation(self.bottom_wall.position),
        ]
    }

    pub fn update(&mut self, delta_time: f32) {
        self.left_paddle.position += Vector3::unit_y() * delta_time * self.left_paddle.velocity;
        self.right_paddle.position += Vector3::unit_y() * delta_time * self.right_paddle.velocity;
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

        }
    }
}
