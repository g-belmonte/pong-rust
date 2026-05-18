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

use engine::camera::{Camera2D, Camera3D};
#[cfg(not(feature = "obj"))]
use engine::graphics_manager::structures::ModelMesh;
use engine::graphics_manager::{
    Binding, DepthMode, GraphicsManager, MaterialDesc, MaterialHandle, VertexAttr,
};
use engine::input::KeyCode;
use engine::resources::{Mesh, Resources};
use engine::scene::{Behaviour, Object, Renderable, Scene, Transform, UpdateCtx};
use engine::{Quat, Vec2, Vec3};

use crate::hud::HudLabelBehaviour;

/// Cube colour (RGB, linear). Modulated per-fragment by Lambertian shading
/// against the lit shader's hardcoded directional light.
const CUBE_COLOR: [f32; 3] = [0.85, 0.45, 0.25];

/// Radians/second around each axis. Picked so the cube spins slowly enough to
/// see the lit faces transition.
const SPIN_X: f32 = TAU * 0.15;
const SPIN_Y: f32 = TAU * 0.22;

/// HUD camera half-height in HUD-world units. With `half_height = 1.0` the
/// vertical axis runs from -1 (top) to +1 (bottom — Camera2D is Y-down) and
/// horizontal runs ±aspect.
const HUD_HALF_HEIGHT: f32 = 1.0;

/// Bundled font for the HUD label. Kept in test-3d/assets so the crate is
/// self-contained — Pong has its own copy.
const HUD_FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");

/// Atlas raster size. 48 px is the same setting Pong's text.rs uses; matched
/// here so visuals are consistent at the resolutions a desktop user runs at.
const HUD_FONT_PX: f32 = 48.0;

/// Atlas-pixel → HUD-world scale. 0.0035 puts the label glyphs at roughly
/// 17% of HUD height — visible but not overwhelming.
const HUD_TEXT_SCALE: f32 = 0.0035;

/// HUD label sits near the top of the screen (Camera2D is Y-down, so
/// negative Y is up).
const HUD_TEXT_POSITION: Vec3 = Vec3::new(0.0, -0.82, 0.0);

const HUD_LABEL: &str = "3D demo (Esc to quit)";

pub fn build_scene(resources: &mut Resources, gm: &mut GraphicsManager) -> Scene {
    let mut scene = Scene::new();

    // Slot 0: 3D world camera (the cube renders against this).
    scene.set_camera(
        0,
        Box::new(Camera3D::new(
            Vec3::new(2.5, 2.0, 4.0),
            Vec3::ZERO,
            Vec3::Y,
            std::f32::consts::FRAC_PI_3,
            0.1,
            100.0,
        )),
    );
    // Slot 1: 2D HUD camera (the text label renders against this).
    scene.set_camera(1, Box::new(Camera2D::new(Vec2::ZERO, HUD_HALF_HEIGHT)));

    let lit_material = resources.load_material(
        gm,
        &MaterialDesc {
            vertex_spv: include_bytes!("../../engine/shaders/spv/lit.vert.spv"),
            fragment_spv: include_bytes!("../../engine/shaders/spv/lit.frag.spv"),
            vertex_attrs: &[VertexAttr::F32x3, VertexAttr::F32x3, VertexAttr::F32x2],
            instance_attrs: &[VertexAttr::Mat4, VertexAttr::F32x3],
            bindings: &[Binding::CameraUbo(0)],
            depth: DepthMode::ReadWrite,
        },
    );

    // Hand-rolled cube by default; the `obj` Cargo feature switches to
    // loading the same geometry from `assets/cube.obj` via the engine's
    // OBJ loader (smoke-test for `Resources::load_obj`).
    #[cfg(not(feature = "obj"))]
    let cube_mesh = resources.load_mesh(gm, &cube_mesh());
    #[cfg(feature = "obj")]
    let cube_mesh = {
        let bytes = include_bytes!("../assets/cube.obj");
        resources
            .load_obj(gm, bytes)
            .into_iter()
            .next()
            .expect("cube.obj produced no meshes")
    };

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

    // HUD: a text label drawn against camera slot 1, with depth disabled so
    // it paints on top of the cube. The label Object has no `Renderable` of
    // its own — `HudLabelBehaviour` owns the per-glyph instances and
    // contributes them via `collect_renderables`.
    let hud_material = crate::hud::register_hud_material(resources, gm);
    let atlas = resources.load_font(gm, HUD_FONT_BYTES, HUD_FONT_PX);
    scene.spawn(
        Object::new()
            .with_position(HUD_TEXT_POSITION)
            .with_behaviour(HudLabelBehaviour::new(
                gm,
                atlas,
                hud_material,
                HUD_LABEL,
                HUD_TEXT_SCALE,
            )),
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
#[cfg(not(feature = "obj"))]
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
