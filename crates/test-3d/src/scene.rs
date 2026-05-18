//! Scene construction for the rotating-cube smoke test.
//!
//! ## What this exercises
//!
//! - A custom material with a 3D vertex layout (`pos: vec3`, `normal: vec3`,
//!   `uv: vec2`), per-instance `mat4` model + `vec3` colour, and
//!   `DepthMode::ReadWrite` so depth-tested 3D opaque geometry renders
//!   correctly.
//! - A hand-rolled cube mesh packed into `ModelMesh`'s byte buffer.
//! - `Camera3D` with perspective projection (Y-up world; Y-flip lives in
//!   `Camera3D::proj`).
//! - A behaviour that integrates `ctx.time.delta_time()` to spin the cube,
//!   exercising the variable-step update hook.

use std::any::Any;
use std::f32::consts::TAU;

use engine::camera::Camera3D;
use engine::graphics_manager::structures::ModelMesh;
use engine::graphics_manager::{
    Binding, DepthMode, GraphicsManager, MaterialDesc, MaterialHandle, VertexAttr,
};
use engine::input::KeyCode;
use engine::resources::{Mesh, Resources};
use engine::scene::{Behaviour, Object, Renderable, Scene, Transform, UpdateCtx};
use engine::{Quat, Vec3};

/// Cube colour (RGB, linear). Modulated per-fragment by Lambertian shading
/// against the lit shader's hardcoded directional light.
const CUBE_COLOR: [f32; 3] = [0.85, 0.45, 0.25];

/// Radians/second around each axis. Picked so the cube spins slowly enough to
/// see the lit faces transition.
const SPIN_X: f32 = TAU * 0.15;
const SPIN_Y: f32 = TAU * 0.22;

pub fn build_scene(resources: &mut Resources, gm: &mut GraphicsManager) -> Scene {
    let mut scene = Scene::new();

    scene.camera = Box::new(Camera3D::new(
        Vec3::new(2.5, 2.0, 4.0),
        Vec3::ZERO,
        Vec3::Y,
        std::f32::consts::FRAC_PI_3,
        0.1,
        100.0,
    ));

    let lit_material = resources.load_material(
        gm,
        &MaterialDesc {
            vertex_spv: include_bytes!("../../engine/shaders/spv/lit.vert.spv"),
            fragment_spv: include_bytes!("../../engine/shaders/spv/lit.frag.spv"),
            vertex_attrs: &[VertexAttr::F32x3, VertexAttr::F32x3, VertexAttr::F32x2],
            instance_attrs: &[VertexAttr::Mat4, VertexAttr::F32x3],
            bindings: &[Binding::CameraUbo],
            depth: DepthMode::ReadWrite,
        },
    );

    let cube_mesh = resources.load_mesh(gm, &cube_mesh());

    scene.spawn(
        Object::new()
            .with_transform(Transform {
                position: Vec3::ZERO,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .with_renderable(Renderable::Material {
                material: lit_material,
                mesh: Some(cube_mesh.handle()),
                texture: None,
                // Per-instance payload after the model matrix = vec3 colour.
                instance_data: f32x3_bytes(CUBE_COLOR),
            })
            .with_behaviour(Spinner {
                _mesh: cube_mesh,
                _material: lit_material,
                yaw: 0.0,
                pitch: 0.0,
            }),
    );

    scene
}

/// Spins the cube in place; quits on Escape.
struct Spinner {
    // Hold the RAII Mesh so the cube's GPU buffers outlive the scene rather
    // than getting queued for destruction immediately after build_scene
    // returns. The material handle is not RAII today (see Resources docs);
    // store it for completeness.
    _mesh: Mesh,
    _material: MaterialHandle,
    yaw: f32,
    pitch: f32,
}

impl Behaviour for Spinner {
    fn update(&mut self, ctx: &mut UpdateCtx) {
        let dt = ctx.time.delta_time();
        self.yaw = (self.yaw + SPIN_Y * dt) % TAU;
        self.pitch = (self.pitch + SPIN_X * dt) % TAU;

        if let Some(obj) = ctx.scene.get_mut(ctx.self_id) {
            obj.transform.rotation =
                Quat::from_axis_angle(Vec3::Y, self.yaw) * Quat::from_axis_angle(Vec3::X, self.pitch);
        }

        if ctx.input.was_just_pressed(KeyCode::Escape) {
            ctx.request_exit();
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Build a unit cube centred at the origin with per-face normals. Vertex
/// layout matches the lit material: `pos: vec3, normal: vec3, uv: vec2` =
/// 32 bytes per vertex. 24 vertices (4 per face × 6 faces), 36 indices.
fn cube_mesh() -> ModelMesh {
    // Six faces, each as (normal, four corner positions in CW winding when
    // looked at from outside, four corner UVs).
    // CW because the engine's pipelines use FrontFace::CLOCKWISE +
    // CullMode::BACK (see graphics_manager::material::create_pipeline).
    let h = 0.5;
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        // +X (right)
        ([1.0, 0.0, 0.0], [[h, -h, h], [h, -h, -h], [h, h, -h], [h, h, h]]),
        // -X (left)
        ([-1.0, 0.0, 0.0], [[-h, -h, -h], [-h, -h, h], [-h, h, h], [-h, h, -h]]),
        // +Y (top, world is Y-up)
        ([0.0, 1.0, 0.0], [[-h, h, h], [h, h, h], [h, h, -h], [-h, h, -h]]),
        // -Y (bottom)
        ([0.0, -1.0, 0.0], [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]]),
        // +Z (front, towards camera at z = 4)
        ([0.0, 0.0, 1.0], [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]]),
        // -Z (back)
        ([0.0, 0.0, -1.0], [[h, -h, -h], [-h, -h, -h], [-h, h, -h], [h, h, -h]]),
    ];
    let face_uvs: [[f32; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

    let mut vertex_bytes = Vec::with_capacity(24 * 32);
    let mut indices = Vec::with_capacity(36);
    for (face_idx, (normal, corners)) in faces.iter().enumerate() {
        let base = (face_idx * 4) as u32;
        for (corner_idx, pos) in corners.iter().enumerate() {
            vertex_bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(pos.as_ptr() as *const u8, 12)
            });
            vertex_bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(normal.as_ptr() as *const u8, 12)
            });
            vertex_bytes.extend_from_slice(unsafe {
                std::slice::from_raw_parts(face_uvs[corner_idx].as_ptr() as *const u8, 8)
            });
        }
        // CW: 0-1-2, 2-3-0 (with the corners ordered CW from outside).
        indices.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 3, base]);
    }

    ModelMesh {
        vertex_bytes,
        vertex_stride: 32,
        indices,
    }
}

fn f32x3_bytes(v: [f32; 3]) -> Vec<u8> {
    let mut out = vec![0u8; 12];
    out.copy_from_slice(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, 12) });
    out
}
