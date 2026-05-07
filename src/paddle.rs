use cgmath::Vector3;

use crate::graphics_manager::structures::{ModelMesh, Vertex};
use crate::graphics_manager::{GraphicsManager, MeshHandle, ModelHandle};

pub struct Paddle {
    pub model_handle: ModelHandle,
    pub position: Vector3<f32>,
    pub velocity: f32,
    pub height: f32,
    pub width: f32,
}

impl Paddle {
    /// Register a shared paddle mesh. Both paddles use the same dimensions today
    /// so callers should register once and reuse the handle for both instances.
    pub fn register_mesh(gm: &mut GraphicsManager, height: f32, width: f32) -> MeshHandle {
        let half_height = height / 2.0;
        let half_width = width / 2.0;
        let mesh = ModelMesh {
            vertices: vec![
                Vertex { pos: [-half_width, -half_height] },
                Vertex { pos: [ half_width, -half_height] },
                Vertex { pos: [ half_width,  half_height] },
                Vertex { pos: [-half_width,  half_height] },
            ],
            indices: vec![0u32, 1, 2, 2, 3, 0],
        };
        gm.register_mesh(&mesh)
    }

    /// Build a paddle that reuses a previously-registered mesh.
    pub fn with_mesh(
        gm: &mut GraphicsManager,
        mesh: MeshHandle,
        position: Vector3<f32>,
        height: f32,
        width: f32,
        color: [f32; 3],
    ) -> Self {
        let model_handle = gm.register_instance(mesh, color);
        Self {
            model_handle,
            position,
            velocity: 0.0,
            height,
            width,
        }
    }
}
