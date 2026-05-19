//! Compile pong's tinted-text material shaders by reusing the engine
//! crate's build helper. Shells out to `glslc` (Arch: `shaderc`). See
//! `crates/engine/build.rs` for the why + failure modes.

#[path = "../engine/build.rs"]
mod engine_build;

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("build.rs: CARGO_MANIFEST_DIR not set"),
    );
    engine_build::compile_shader_dir(
        &manifest_dir.join("shaders/src"),
        &manifest_dir.join("shaders/spv"),
    );
}
