use cgmath::Matrix4;

pub struct Camera {
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}

impl Camera {
    pub fn new(view: Matrix4<f32>, proj: Matrix4<f32>) -> Self {
        Self { view, proj }
    }
}
