use cgmath::{Matrix4, Point3, Vector3, Deg, SquareMatrix};

use crate::camera::Camera;
use crate::graphics_manager::constants::{WINDOW_WIDTH, WINDOW_HEIGHT};
use crate::paddle::Paddle;

pub struct Scene {
    pub camera: Camera,
    pub left_paddle: Paddle,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            camera: Camera::new(
                Matrix4::look_at(
                    Point3::new(0.0, 0.0, 4.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0)
                ),
                cgmath::perspective(
                    Deg(45.0),
                    WINDOW_WIDTH as f32 / WINDOW_HEIGHT as f32,
                    0.1,
                    10.0,
                ),
            ),
            left_paddle: Paddle::new(Matrix4::<f32>::identity()),
        }
    }
}
