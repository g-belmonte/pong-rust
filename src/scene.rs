use cgmath::{Deg, Matrix4, Point3, SquareMatrix, Vector3};

use crate::camera::Camera;
use crate::graphics_manager::constants::{WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::paddle::{ModelMesh, Paddle};

pub struct ModelData {
    pub model_mesh: ModelMesh,
    pub model_transform: Matrix4<f32>,
}

pub struct Scene {
    pub camera: Camera,
    pub left_paddle: Paddle,
    pub right_paddle: Paddle,
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
            left_paddle: Paddle::new(
                Matrix4::<f32>::identity()
                    + Matrix4::from_translation(Vector3 {
                        x: -3.7,
                        y: 0.0,
                        z: 0.0,
                    }),
            ),
            right_paddle: Paddle::new(
                Matrix4::<f32>::identity()
                    + Matrix4::from_translation(Vector3 {
                        x: 3.7,
                        y: 0.0,
                        z: 0.0,
                    }),
            ),
        }
    }

    pub fn get_model_data(&self) -> Vec<ModelData> {
        vec![
            ModelData {
                model_mesh: self.left_paddle.model_mesh.clone(),
                model_transform: self.left_paddle.model_transform,
            },
            ModelData {
                model_mesh: self.right_paddle.model_mesh.clone(),
                model_transform: self.right_paddle.model_transform,
            },
        ]
    }

    pub fn get_model_transforms(&self) -> Vec<Matrix4<f32>> {
        vec![
            self.left_paddle.model_transform,
            self.right_paddle.model_transform,
        ]
    }
}
