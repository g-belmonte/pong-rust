# CLAUDE.md

Guidance for Claude Code when working in this repo. **Start by reading [ARCHITECTURE.md](ARCHITECTURE.md)** for the engine's vision, subsystems, and load-bearing design decisions. This file only carries what's not in there and not easily inferred from the code itself: build commands, external requirements, ash-version gotchas, and cross-file invariants that wouldn't be obvious from reading any one file.

## Build & run

```sh
cargo build                                                                       # workspace (build.rs auto-compiles shaders)
cargo run -p pong                                                                 # opens on the main menu
cargo run -p pong --release
cargo run -p test-3d                                                              # 3D smoke test (Escape quits)
cargo run -p pong --features hot-reload                                           # opt-in shader + asset hot-reload
```

Shaders are compiled by `crates/engine/build.rs` + `crates/pong/build.rs` (both shell out to `glslc`, Arch: `shaderc`). The engine still `include_bytes!`'s the SPV blobs, so the build.rs writes them under `shaders/spv/` before the lib crate compiles. Cargo's `rerun-if-changed=shaders/src` makes the script re-run on any GLSL edit. A missing `glslc` fails the build with a clear panic from build.rs — much earlier than the old `include_bytes!` not-found error.

`scripts/compile-shaders.sh` is kept around as a one-shot manual compile (useful as an editor on-save hook in the `hot-reload` workflow, where the running binary needs SPV updates without a full `cargo build`).

The "delete SPV dir to force rebuild" trick from before *won't* trigger a rebuild — cargo doesn't track build-script outputs, so missing SPV outputs aren't a signal for cargo to re-run build.rs. If you really need a clean slate, `cargo clean -p engine -p pong` or `touch crates/engine/shaders/src/main.vert` instead.

External runtime requirements: a Vulkan-capable GPU + drivers, plus Khronos validation layers (Arch: `vulkan-validation-layers`). `VALIDATION.is_enable = true` in `crates/engine/src/graphics_manager/constants.rs` is hard-coded — the binary fails to create the Vulkan instance if `VK_LAYER_KHRONOS_validation` is not installed. No test suite; `cargo test` runs nothing.

User settings persist as JSON at `dirs::config_dir()/pong-rust/settings.json`. Save timing: on Escape from Settings, on Escape from menu, on menu Quit — never in-game. Missing or garbage file = defaults; values are clamped on read.

## Engine scope (guidance for Claude)

The engine is 2D + 3D capable. **Pong is intentionally 2D** — follow the Y-down convention there, prefer `Renderable::Solid` / `Renderable::Textured` or pong-side custom materials like `text_tint`, and don't pull in 3D paths (`Camera3D`, depth-aware materials, lit shaders) without a concrete in-game reason. Engine-side work can target either path.

## Asset references

Non-shader asset call sites in pong + test-3d go through the `engine::asset!("relative/path/from/crate/root")` macro. In release it expands to `AssetSource::from_bytes(include_bytes!(...))` (assets baked into the binary, zero runtime cost). With `--features hot-reload` it expands to `AssetSource::from_file(...)` which reads at runtime and registers the path with the asset watcher. Loaders that take `AssetSource`: `load_texture_png`, `load_sound`, `load_font`, `load_obj`, `load_gltf`. **Don't reintroduce `include_bytes!` directly at call sites** — use the macro so the dev/release switch keeps working. The `concat!(env!("CARGO_MANIFEST_DIR"), "/", $path)` inside the macro is resolved at the call site, so `CARGO_MANIFEST_DIR` correctly anchors at the game crate even though the macro lives in `engine`.

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
- **Shared `Mesh` / `FontAtlas` RAII wrappers must live on a behaviour that outlives every instance built against them.** Pong's game scene parks the paddle/wall/digit meshes + font atlas on `PhaseController`, and the dimmer mesh + a font-atlas clone on `PauseController`. Drop them at the end of the scene-builder and `flush_pending` will destroy the GPU resource between the next `apply_commands` and `draw_frame`, panicking the renderer's `self.meshes[&handle]` lookup. The `_` prefix on these fields (`_paddle_mesh`, `_dimmer_mesh`, `_atlas`) flags "held for lifetime extension only, not read directly".
- **The "always emit, park hidden ones at `hidden_transform()`" rule for multi-instance behaviours.** `register_instance` initialises `last_model` to identity (= world origin), not off-screen — silently skipping hidden segments/glyphs from `collect_renderables` would leave them parked at the origin until the next emit.
- **`Object::renderable` is registered by the engine on `Scene::apply_commands`, not by the caller.** Game code only supplies the `Renderable` variant + handles. This is what makes `Scene::despawn(id)` work without the caller remembering the handle.
- **`reload_texture` keeps the `TextureHandle` slot stable but rebuilds image+memory+view and re-binds every material's sampler descriptor sets that referenced the old view.** Any future code path that caches `vk::ImageView` outside `TextureResources` would silently keep the old view after a hot-reload — re-derive from `self.textures[&handle].view` each draw, never stash.
- **Ball `MAX_SPEED` in `crates/pong/src/ball.rs` is bound to the no-substep invariant.** It must satisfy `MAX_SPEED * fixed_dt < PADDLE_WIDTH` per axis (the assumption documented at `ball.rs` `fixed_update`). If you change `PADDLE_WIDTH` in `scene_game.rs` or the engine's 120 Hz fixed step, re-check the cap — otherwise a fast ball can tunnel through a paddle.
- **Pause-overlay layering is registration-order-sensitive.** In `scene_game::build_game`, the dimmer material *must* register before `register_tinted_text_material(...)` — both run `DepthMode::Disabled`, so material registration order is paint order. Reversing them paints the backdrop over its own menu text.
- **Only `PauseController` owns ESC in the game scene.** `PhaseController::update` early-returns when `ctx.scene.is_paused()` so SPACE can't drive phase transitions under the overlay, and ESC handling lives exclusively on `PauseController`. If you add another in-game behaviour that consumes keyboard edges in `update`, gate it the same way.

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
- **"press ESC to return" on Settings is at a fixed world position; can clip on very narrow aspect ratios.** Fix: per-frame "track the bottom-right corner" behaviour using `Camera2D` + `GraphicsManager::extent`.
