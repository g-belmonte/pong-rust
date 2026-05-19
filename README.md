# Pong-rust

Implementation of the game Pong using rust as programming language.

For this project I followed the "vulkan tutorial" up to the index buffers part using `unkownue/vulkan-tutorial-rust` repository as inspiration.

The project is structured in three layers:

- **`engine`** — a reusable Rust game-engine library (Vulkan renderer, audio, resources, scene + behaviours, input, time, 2D + 3D cameras, depth-aware material pipelines).
- **`pong`** — a thin application crate that uses `engine` to implement the game.
- **`test-3d`** — a tiny smoke-test crate exercising the engine's 3D path: depth buffer, perspective camera, custom 3D lit material (`pos/normal/uv` vertex layout), plus a 2D HUD text label drawn against a second camera slot — the canonical "3D world + 2D HUD" mixed-camera demo.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design. See [CLAUDE.md](CLAUDE.md) for the current state of the code.

# Getting started

## External dependencies

This project was conceived in an Archlinux system which already had a lot of development tools available, so some external dependencies might have been overlooked or simply ignored. Below follows a list with some of the dependencies deemed important to work with this code.

### Graphics card and drivers

This project runs using the Vulkan API, so it is expected that its host would kindly provide a graphics card with Vulkan support and have all required drivers in proper working condition.

### Shader compiler

The binary `glslc` is used to compile GLSL into SPIR-V or something of the sorts.

Arch package: [shaderc](https://archlinux.org/packages/extra/x86_64/shaderc/)

### Vulkan stuff

Arch packages:
  - [vulkan-tools](https://archlinux.org/packages/extra/x86_64/vulkan-tools/)
  - [vulkan-extra-tools](https://archlinux.org/packages/extra/x86_64/vulkan-extra-tools/)
  - [vulkan-validation-layers](https://archlinux.org/packages/extra/x86_64/vulkan-validation-layers/)

### Rust

Tested with `rustc 1.73.0`

## Compile shaders

Shaders are compiled automatically by each crate's `build.rs` on `cargo
build` — it shells out to `glslc` (Arch: `shaderc`) for every file in
`shaders/src/` and writes the SPV to `shaders/spv/`. Cargo re-runs the
build script when any source under `shaders/src/` changes, so a normal
edit-build cycle picks up shader changes without extra steps.

Manual one-shot compile (e.g. as an editor on-save hook for the
`hot-reload` workflow described below) is still available via the legacy
script: `scripts/compile-shaders.sh crates/engine/shaders/src crates/engine/shaders/spv`.

## Controls

The game opens on a main menu with **Play**, **Settings**, and **Quit**.
Navigate with the mouse (hover to highlight, left-click to select) or with
**Up / Down + Enter**. **Escape** quits from the menu.

On the **Settings** screen: **Up / Down** moves between rows, **Left / Right**
adjusts the highlighted setting (winning score 1–9, ball speed 1.0–8.0,
paddle speed 0.5–4.0; speeds step by 0.5), **Escape** saves to disk and
returns to the menu. Settings persist as JSON at
`$XDG_CONFIG_HOME/pong-rust/settings.json` (or the platform equivalent —
`~/Library/Application Support/...` on macOS, `%APPDATA%\...` on Windows)
and load automatically on the next launch.

In-game: **W / S** drive the left paddle, **I / K** drive the right paddle,
**Space** starts a match (and replays after Game Over), **Escape** returns to
the main menu. Settings changes take effect at the *next* Play — a match in
progress keeps its starting values.

## Compile and run the game

### Debug/Dev profile

`cargo build` (workspace) and `cargo run -p pong`.

To run the 3D smoke-test instead: `cargo run -p test-3d` (one lit rotating cube under a 2D HUD label; Escape quits).

The test-3d crate also has `obj` / `gltf` feature proxies that turn on the matching engine loader; running `cargo run -p test-3d --features obj` loads the cube from `crates/test-3d/assets/cube.obj` instead of the hand-rolled vertex bytes, exercising the engine's OBJ loader end-to-end.

### Release profile

Same as debug, but with a `--release` flag added to the listed commands.

### Hot-reload (dev iteration)

The `hot-reload` Cargo feature covers **both** shaders and game-side
assets. Off by default; release builds get zero overhead (assets are baked
into the binary via `include_bytes!`, the watcher type is `#[cfg]`-gated out).

```sh
cargo run -p pong --features hot-reload
cargo run -p test-3d --features hot-reload,obj   # exercises OBJ mesh reload
```

**Shaders.** The engine watches `crates/engine/shaders/spv/` and rebuilds
the affected built-in material's pipeline whenever SPV bytes change on disk.
Workflow: edit a GLSL file under `crates/engine/shaders/src/`, run
`scripts/compile-shaders.sh crates/engine/shaders/src crates/engine/shaders/spv`
(or hook it into your editor's on-save). Bad SPV bytes (e.g. a compile error
producing a broken file) are logged to stderr and the old pipeline keeps
running.

**Non-shader assets.** The engine also watches each game crate's `assets/`
directory and, on file change, applies the right kind of in-place update:

| Asset kind          | Reload behaviour                                                              |
|---------------------|-------------------------------------------------------------------------------|
| **PNG texture**     | Re-decode + in-place GPU update; descriptor sets re-bound to the new image.   |
| **OBJ / glTF mesh** | Re-parse + in-place GPU update on every affected `MeshHandle`.                |
| **MP3 / sound**     | Re-decode + swap inside the `Sound` cell; every clone picks it up on `play`.  |
| **TTF font**        | Log-only (`restart to apply`). Re-baking can shift glyph metrics and break already-laid-out text. |

Asset references in game code go through the `engine::asset!("relative/path")`
macro: in release it expands to `include_bytes!`, in dev it reads from disk
and registers the source path with the watcher. The same call site works
in both modes.

# Open items

- Game modes (single player vs CPU, best-of-N matches, etc.).
