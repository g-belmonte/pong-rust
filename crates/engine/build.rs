//! Compile every GLSL source under `shaders/src/` to SPIR-V under
//! `shaders/spv/` by shelling out to `glslc` (Arch: `shaderc` package).
//!
//! Runs on every `cargo build` of the engine crate; Cargo re-runs us
//! whenever a file under `shaders/src/` changes (or this build.rs itself).
//! Source-of-truth is the GLSL on disk; the SPV files are derived artifacts
//! and the `include_bytes!` sites in `graphics_manager.rs` reference them
//! directly. A missing `glslc` or a GLSL compile error fails the build
//! loudly — the old behaviour was a silent stale SPV until next runtime.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;

// `#[allow(dead_code)]` on `main` is for pong's `build.rs`, which imports
// this file via `#[path = "../engine/build.rs"] mod engine_build;` and only
// uses `compile_shader_dir` — pong has its own `main`.
#[allow(dead_code)]
fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("build.rs: CARGO_MANIFEST_DIR not set"),
    );
    let src_dir = manifest_dir.join("shaders/src");
    let spv_dir = manifest_dir.join("shaders/spv");
    compile_shader_dir(&src_dir, &spv_dir);
}

/// Compile every file directly under `src_dir` to `<spv_dir>/<filename>.spv`
/// by shelling out to `glslc`. Public so pong's `build.rs` can reuse it via
/// `#[path = "../engine/build.rs"] mod engine_build;`.
///
/// Format-arg inlining (`"{x}"`) is deliberately avoided — pong is on edition
/// 2018 and reuses this file via `#[path]`, so it has to lint cleanly under
/// both editions.
pub fn compile_shader_dir(src_dir: &Path, spv_dir: &Path) {
    println!("cargo:rerun-if-changed={}", src_dir.display());
    // Re-run if the build script itself changes (e.g. flag tweaks).
    println!("cargo:rerun-if-changed=build.rs");

    if !src_dir.is_dir() {
        panic!(
            "build.rs: shader source dir does not exist: {}",
            src_dir.display()
        );
    }

    // Verify glslc is on PATH up-front so the error message is clear.
    if Command::new("glslc").arg("--version").output().is_err() {
        panic!(
            "build.rs: `glslc` not found on PATH. Install the shader compiler (Arch: `shaderc`) \
             and rebuild. The SPV outputs under {} are derived from {} on every build.",
            spv_dir.display(),
            src_dir.display(),
        );
    }

    if let Err(e) = std::fs::create_dir_all(spv_dir) {
        panic!("build.rs: create_dir_all {} failed: {}", spv_dir.display(), e);
    }

    let entries = match std::fs::read_dir(src_dir) {
        Ok(it) => it,
        Err(e) => panic!("build.rs: read_dir {} failed: {}", src_dir.display(), e),
    };

    for entry in entries {
        let entry = entry.expect("build.rs: read_dir entry failed");
        let src_path = entry.path();
        if !src_path.is_file() {
            continue;
        }
        // Skip backup/swap files some editors drop next to the originals.
        if let Some(name) = src_path.file_name().and_then(OsStr::to_str) {
            if name.starts_with('.') || name.ends_with('~') {
                continue;
            }
        }
        let file_name = src_path
            .file_name()
            .expect("build.rs: shader source has no file name");
        let dst_name = {
            let mut s = file_name.to_os_string();
            s.push(".spv");
            s
        };
        let dst_path = spv_dir.join(&dst_name);

        let status = match Command::new("glslc")
            .arg(&src_path)
            .arg("-o")
            .arg(&dst_path)
            .status()
        {
            Ok(s) => s,
            Err(e) => panic!("build.rs: failed to invoke glslc: {}", e),
        };
        if !status.success() {
            panic!(
                "build.rs: glslc failed for {} (exit {:?})",
                src_path.display(),
                status.code()
            );
        }
    }
}
