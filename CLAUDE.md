# CLAUDE.md

Guidance for Claude Code when working in this repo. **Start by reading [ARCHITECTURE.md](ARCHITECTURE.md)** for the engine's vision, subsystems, and load-bearing design decisions. This file only carries what's not in there and not easily inferred from the code itself: build commands, external requirements, ash-version gotchas, and cross-file invariants that wouldn't be obvious from reading any one file.

## Build & run

```sh
scripts/compile-shaders.sh crates/engine/shaders/src crates/engine/shaders/spv   # requires `glslc` (Arch: shaderc)
scripts/compile-shaders.sh crates/pong/shaders/src crates/pong/shaders/spv       # pong's tinted-text material (menu)
cargo build                                                                       # workspace
cargo run -p pong                                                                 # opens on the main menu
cargo run -p pong --release
cargo run -p test-3d                                                              # 3D smoke test (Escape quits)
cargo run -p pong --features hot-reload                                           # opt-in shader hot-reload
```

Shaders must be compiled **before** `cargo build` — the engine `include_bytes!`'s the SPV blobs, so a missing `crates/engine/shaders/spv/` directory is a compile error, not a runtime one. The shader script doesn't actually clean stale outputs (the `rm -rf "$output_folder/*"` is unquoted and matches nothing) — when in doubt, `rm -rf crates/engine/shaders/spv crates/pong/shaders/spv` first.

External runtime requirements: a Vulkan-capable GPU + drivers, plus Khronos validation layers (Arch: `vulkan-validation-layers`). `VALIDATION.is_enable = true` in `crates/engine/src/graphics_manager/constants.rs` is hard-coded — the binary fails to create the Vulkan instance if `VK_LAYER_KHRONOS_validation` is not installed. No test suite; `cargo test` runs nothing.

User settings persist as JSON at `dirs::config_dir()/pong-rust/settings.json`. Save timing: on Escape from Settings, on Escape from menu, on menu Quit — never in-game. Missing or garbage file = defaults; values are clamped on read.

## Engine scope (guidance for Claude)

The engine is 2D + 3D capable. **Pong is intentionally 2D** — follow the Y-down convention there, prefer `Renderable::Solid` / `Renderable::Textured` or pong-side custom materials like `text_tint`, and don't pull in 3D paths (`Camera3D`, depth-aware materials, lit shaders) without a concrete in-game reason. Engine-side work can target either path.

## Workspace particulars

- Engine and test-3d are **edition 2021**; pong is still **edition 2018**. Don't assume workspace-wide edition.
- glam is pinned in `crates/engine/Cargo.toml` (`glam = "0.32"`) and re-exported via `engine::{Mat4, Quat, Vec2, Vec3, Vec4}`. Pong and test-3d have no direct `glam` dep and shouldn't gain one.

## Ash 0.38 / Vulkan 1.3.281 specifics

Non-obvious if your last `ash` was 0.29-ish or earlier.

- Extension loaders are under `ash::khr::{surface, swapchain}::{Instance, Device}` and `ash::ext::debug_utils::Instance` (one struct per extension; suffix = which Vulkan handle it wraps).
- `vk::*CreateInfo` structs carry a `_marker: PhantomData<&'a ()>` for the `p_next`-chain lifetime. Struct literals end with `..Default::default()` to fill it — **except** `vk::ClearValue`, which is a `union` and can't use functional-update syntax.
- Builders auto-deref to their struct. The idiom is `vk::Foo::default().a(x).b(y)` — `.build()` is gone.
- Per-device validation layers are deprecated and ignored by modern loaders; don't wire them up.
- `vk::ColorComponentFlags::all()` is gone; use `vk::ColorComponentFlags::RGBA`.
- Surface creation goes through `ash-window`; winit 0.30 exposes `raw-window-handle` 0.6 natively, so there is no per-OS `#[cfg]` plumbing.

## Cross-file invariants (won't catch your eye reading one file)

- **`CAMERA_HALF_HEIGHT` is duplicated** in `scene_game.rs`, `scene_menu.rs`, and `scene_settings.rs` so all three scenes share the same camera and swaps don't re-letterbox. Change one, change all three.
- **`scene_game::build_game` spawn order matters**: walls first (so paddles + ball can capture their IDs), then paddles + ball, then `PhaseController` last (captures every other ID + a clone of the Settings Rc). Re-ordering this breaks ID wiring silently at runtime.
- **`Renderable::Material::instance_data` length must equal `instance_stride - 64`** (asserted at register; the engine writes the leading `Mat4` itself). If you change a material's `instance_attrs`, every caller packing its bytes has to change too.
- **The first instance attribute of every material must be `Mat4`** (asserted at `register_material`). The engine writes the per-instance model matrix at offset 0.
- **Editing a built-in material's shader requires three changes in lockstep**: the GLSL source, the `vertex_attrs`/`instance_attrs` list in `GraphicsManager::new`, and any pong-side code packing matching instance bytes.
- **Material registration order = paint order** for the depth-disabled built-ins. Custom materials registered *after* the built-ins paint on top. Pong's `register_tinted_text_material` is re-registered per scene that uses it (no `load_material_cached` API today); accumulation is bounded by distinct (scene, material) pairs.
- **`FontAtlas` lives on `PhaseController`** in pong's game scene — the longest-lived behaviour — so its `Texture` outlives every glyph instance registered against it. If you move it, glyph instances may outlive the atlas and the unregister path will run against a freed handle.
- **The "always emit, park hidden ones at `hidden_transform()`" rule for multi-instance behaviours.** `register_instance` initialises `last_model` to identity (= world origin), not off-screen — silently skipping hidden segments/glyphs from `collect_renderables` would leave them parked at the origin until the next emit.
- **`Object::renderable` is registered by the engine on `Scene::apply_commands`, not by the caller.** Game code only supplies the `Renderable` variant + handles. This is what makes `Scene::despawn(id)` work without the caller remembering the handle.

## Timing invariants

- **`Resources::flush_pending(&mut gm)` runs once per frame, right before `draw_frame`.** Drop pushes the id; the actual `device_wait_idle` + `unregister_*` happens at this known-safe boundary. Don't try to free directly from `Drop` — would need an unsafe back-pointer to the renderer and could race the GPU. Also: don't move the flush into `Drop` — the engine bridges winit 0.30's `Result`-returning `run_app` via `process::exit` to keep `App::run() -> !`, so App's locals don't drop on graceful exit.
- **Pipeline cache is saved at end of `GraphicsManager::new()`** (disk path `dirs::data_dir()/pong-rust/pipeline.cache`). Same `process::exit` issue — the end-of-init save is the load-bearing one; the `Drop` save is belt-and-braces.
- **`Scene::apply_commands(&mut gm)` runs after `build_scene`, after every `dispatch_fixed_update` substep, and after `dispatch_update`.** Spawning from `fixed_update` is observed in the *next* substep, not partway through the frame. The post-update flush is *before* `flush_pending`, so RAII drops triggered by `on_despawn` queue and flush in the same frame.
- **Camera matrices are cache-checked per slot.** Moving cameras pay `device_wait_idle` per frame today (deferred fix: per-frame-in-flight camera UBOs). `recreate_swapchain` invalidates `last_camera_*` so the next `set_camera` writes against the new aspect.
- **`recreate_swapchain` doesn't compute a projection.** Aspect is derived inside `set_camera` from `swapchain_extent`. The render pass rebuilds only when the swapchain *format* changes; descriptor-set layouts, descriptor pool, all descriptor sets, the unit quad, and per-material instance buffers survive across recreations. Pipelines rebuild every recreation (viewport/scissor baked in).

## History / preprocessing notes

These won't be obvious from the code:

- **`tennis-ball.png` was preprocessed from a JPG with a circular-disc mask** — fit the bounding-box centre + radius of non-background pixels, then set alpha = 0 outside the disc. Flood-fill couldn't separate the disc from the corner background because the ball's anti-aliased silhouette connects them.
- **`MenuController` inlines `screen_to_world` against cached `camera_centre`/`camera_half_height`** rather than calling `Camera2D::screen_to_world` because `Scene::camera(0)` returns `Box<dyn Camera>` and doesn't expose `Camera2D`'s fields. Adding a typed downcast path to `Scene` was the alternative; caching the params on the controller was cheaper.
- **Pong currently uses the same MP3 for all four SFX** (`wall_bounce_sfx`, `paddle_bounce_sfx`, `score_sfx`, `game_over_sfx`). The structural slots are separate so swapping in distinct samples is a path-only change in `crates/pong/src/scene_game.rs`.

## Future improvements

- **Game modes** (single-player vs CPU, best-of-N matches, etc.) — the only deliberate gap in pong's current feature set.
- **Filesystem-based asset paths + hot-reload for non-shader assets.** Everything is `include_bytes!`'d today.
- **"press ESC to return" on Settings is at a fixed world position; can clip on very narrow aspect ratios.** Fix: per-frame "track the bottom-right corner" behaviour using `Camera2D` + `GraphicsManager::extent`.
