//! Phase 8 smoke test: a single rotating 3D cube.
//!
//! Demonstrates the engine's 3D path end-to-end — depth buffer, a custom
//! lit material, a 3D cube mesh, and `Camera3D` — without any of the 2D
//! built-ins.

mod scene;

fn main() {
    engine::app::App::new().with_scene(scene::build_scene).run()
}
