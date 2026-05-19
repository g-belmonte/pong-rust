# Architecture

This document captures the **target** architecture of the project and the load-bearing decisions behind it. For the **current** state of the code (modules, signatures, conventions), see `CLAUDE.md`. For build/run instructions, see `README.md`.

## Vision

Two layers, sharply separated:

1. **`engine`** — a reusable 2D + 3D Rust game-engine library. Vulkan renderer, audio, resources, scene + behaviours, input, time, cameras, depth-aware material pipelines.
2. **`pong`** — a thin application crate that uses `engine` to implement the game.

The engine started 2D-first; 3D landed once a concrete in-workspace consumer (`crates/test-3d`) needed it. The engine is now 2D + 3D capable; pong itself stays 2D.

Non-goals (explicit, to avoid drift):

- A general-purpose AAA-grade engine. Bevy already exists.
- A scripting language. Behaviours are written in Rust.
- An editor / scene file format. Scenes are constructed in code.

## Workspace layout

```
pong-rust/
  Cargo.toml                # [workspace] only — no [package]
  ARCHITECTURE.md
  CLAUDE.md
  README.md
  crates/
    engine/
      Cargo.toml
      src/
        lib.rs              # public surface; re-exports submodules + glam
        app.rs              # winit ApplicationHandler + per-frame loop
        audio.rs            # kira wrapper
        camera.rs           # Camera trait, Camera2D, Camera3D
        graphics_manager.rs # renderer entry; submodules under graphics_manager/
        input.rs            # polling Input
        resources/          # mesh/texture/sound/font/material loaders, RAII handles
        scene.rs            # Scene + Object + Behaviour trait + command queue
        time.rs             # delta_time, elapsed, fixed-timestep accumulator
      shaders/{src,spv}/    # engine-owned shaders (main, textured)
    pong/
      Cargo.toml
      src/{main, scene_menu, scene_settings, scene_game, phase, ball,
           paddle, wall, digit, text, tinted_text, settings}.rs
      shaders/{src,spv}/    # pong-owned tinted-text material
      assets/               # tennis-ball.png, DejaVuSans.ttf, sound
    test-3d/
      Cargo.toml
      src/{main, hud, cube}.rs
      assets/cube.obj
  scripts/compile-shaders.sh
```

The repo root `Cargo.toml` is `[workspace] members = ["crates/engine", "crates/pong", "crates/test-3d"]`. Future games drop in as additional crates next to `pong/`.

## Engine subsystems

### `engine::graphics_manager`

Hand-rolled Vulkan renderer via `ash` 0.38 (Vulkan 1.3.281), derived from the tutorial scaffolding in `unknownue/vulkan-tutorial-rust` up through texture mapping. Public API is handle-based (`MeshHandle`, `TextureHandle`, `ModelHandle`, `MaterialHandle`); the registration entry points (`register_mesh` / `register_texture*`) are `pub(crate)` and only `Resources` calls them. Camera matrices come in via `set_camera(slot, &dyn Camera)`; per-slot UBOs are cache-checked so a static camera pays nothing per frame.

The renderer is built around a generic **material registry** — there is no hardcoded "solid pipeline + textured pipeline" split. `MaterialDesc { vertex_spv, fragment_spv, vertex_attrs, instance_attrs, bindings, depth }` is the data-driven descriptor; game code calls `Resources::load_material` and gets a `MaterialHandle`. The engine ships two built-ins (solid + textured) registered the same way as game-side custom materials.

The render pass always carries a `D32_SFLOAT` depth attachment; each material's `DepthMode { Disabled | ReadOnly | ReadWrite }` picks how its pipeline interacts with depth. Mixing modes in one scene is the supported path for "3D world with 2D HUD on top". The built-in materials run `Disabled` so Pong's pre-3D paint-order layering still works.

Each material also declares a `BlendMode { Opaque | Alpha }`. `Opaque` is the default and disables blending; the engine's built-ins and every alpha-discard fragment shader (textured atlas, tinted text) stay there. `Alpha` enables straight (non-premultiplied) source-over blending — used by pong's pause-menu dimmer to render a semi-transparent backdrop over the paused gameplay frame.

Optional `hot-reload` Cargo feature watches `crates/engine/shaders/spv/` and rebuilds the affected built-in material's pipeline on disk change, with `catch_unwind` + SPV rollback so a malformed shader keeps the old pipeline running. The same feature also enables non-shader asset hot-reload through `engine::resources::asset_watcher` (see the *Resources* section). Release builds have zero overhead.

### `engine::resources`

Single API game code calls for asset loading. Loaders take `&mut GraphicsManager` per call (the renderer is owned by `App`, not by `Resources`). Returns:

- `Mesh` / `Texture` — `Rc`-backed clonable handles around `MeshHandle` / `TextureHandle`. Cloning is a refcount bump and shares the same GPU resource. `Resources` keeps a content-addressed `Weak` cache (hashed input → cached `Rc`), so two `load_texture_png(SAME_BYTES)` calls share GPU memory transparently. GPU destruction queues only when the last clone drops.
- `Sound` — kira `Arc`-backed value type; no GPU coupling, no deferred-destroy queue.
- `FontAtlas` — `fontdue` shelf-packed ASCII glyph atlas (512×512 RGBA8). Clonable; the inner `Texture` is refcounted.
- `MaterialHandle` — plain `Copy` id; materials live for the renderer's lifetime (no destruction path yet).

GPU resource destruction goes through an `Rc<RefCell<PendingDestroys>>` queue: `Drop` on the wrapper pushes an id, `Resources::flush_pending(&mut gm)` drains the queue once per frame at a known-safe boundary (where `device_wait_idle` + `unregister_*` is correct).

OBJ + glTF mesh loaders gated behind separate Cargo features (`obj` via `tobj`; `gltf` via `gltf` — `.glb` + embedded data URIs only). Each provides a low-level `*_data` variant returning typed `Vec<MeshData>` and a high-level `load_*` variant that packs into the lit vertex layout (`pos+normal+uv`, stride 32) and returns `Vec<Mesh>`.

#### Asset paths + hot-reload

Asset references in game code go through the `engine::asset!("relative/path")` macro:

- **Release** — expands to `AssetSource::from_bytes(include_bytes!(...))`. Assets are baked into the binary; no runtime filesystem dependency; the watcher type is `#[cfg]`-gated out.
- **Dev (`hot-reload` feature on)** — expands to `AssetSource::from_file(...)` which reads the file at runtime and records the absolute path on the returned `AssetSource`. Loaders that take `AssetSource` (`load_texture_png`, `load_sound`, `load_font`, `load_obj`, `load_gltf`) register the path with `engine::resources::asset_watcher::AssetWatcher`.

The watcher watches each parent directory non-recursively, debounces events at 150 ms, and on a real change re-reads the file and applies the kind-specific reload:

- **PNG texture** → `GraphicsManager::reload_texture(handle, bytes)`: re-decode, in-place rebuild of the GPU image + memory + view backing the same `TextureHandle` slot, descriptor sets for every material sampling the old view are freed and re-allocated against the new one. Image dimensions may change.
- **OBJ / glTF mesh** → re-parse, re-pack into the lit vertex layout, `GraphicsManager::reload_mesh(handle, &ModelMesh)` per sub-mesh. If the sub-mesh count differs from the original load, the mismatch is logged and only the overlapping prefix is updated (restart for the rest).
- **MP3 / sound** → re-decode and swap the data inside every still-alive `Sound` clone's `Rc<RefCell<StaticSoundData>>`. Every behaviour holding a clone picks up the new sample on its next `audio.play(&sound)` call.
- **TTF font** → log-only. Re-baking the atlas can shift glyph metrics, which would leave previously-laid-out text labels positioned against stale `GlyphInfo` and stale UVs. Restart is the prescribed path.

The watcher entries hold `Weak`s, so a path's reload work short-circuits the moment no game code holds a clone of the asset; the entry GCs out on the next `flush_pending`. Failure modes (notify backend error, IO error, PNG decode error, Vulkan upload failure) all log and leave the previously-loaded asset in place — the dev loop keeps running.

### `engine::scene`

Flat `HashMap<ObjectId, Object>` with trait-based behaviours and a deferred command queue.

```rust
// Sketch — see crates/engine/src/scene.rs for actual signatures.
pub struct Object {
    pub transform: Transform,
    pub visible: bool,
    pub renderable: Option<Renderable>,    // Solid | Textured | Material { instance_data: Vec<u8>, ... }
    pub behaviours: Vec<Box<dyn Behaviour>>,
    pub(crate) model_handle: Option<ModelHandle>,
}

pub trait Behaviour: Any {
    fn update(&mut self, ctx: &mut UpdateCtx) {}
    fn fixed_update(&mut self, ctx: &mut UpdateCtx) {}
    fn collect_renderables(&self, parent: Mat4, out: &mut Vec<(ModelHandle, Mat4)>) {}
    fn on_despawn(&mut self, gm: &mut GraphicsManager) {}
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
```

Key shapes:

- **Two-hook dispatch.** `fixed_update` runs at 120 Hz from a fixed-step accumulator (chosen because the step ~8.3 ms is comfortably smaller than the smallest collision feature in Pong, so the substep loop collapses to one integrate-and-resolve pass per tick); `update` runs once per frame with variable delta. `Time::delta_time()` is phase-aware: returns the fixed step inside `fixed_update`, the variable delta inside `update`.
- **Deferred spawn/despawn.** `Scene::spawn(Object) -> ObjectId` pre-allocates the id (so callers can wire references at build time) and queues the insert; `apply_commands(&mut gm)` drains the queue at every dispatch boundary, registers/unregisters renderable instances, and runs `on_despawn` hooks.
- **Cross-behaviour mutation via typed downcast.** `Scene::behaviour::<T>(id)` / `behaviour_mut::<T>(id)` use `Any` + `TypeId` to reach a sibling object's behaviour. Each behaviour must one-line-impl `as_any` / `as_any_mut`.
- **Multi-instance entities are one Object whose behaviour owns extra `ModelHandle`s.** Digit segments (7), text glyphs (N), etc. The Object's `renderable` is `None`; the behaviour contributes `(handle, parent_matrix * local)` pairs from `collect_renderables` and unregisters them in `on_despawn`. Visibility uses the "always emit, park hidden ones at `hidden_transform()`" trick (necessary because `register_instance` initialises `last_model` to identity, not off-screen).
- **Cameras live on `Scene::cameras: HashMap<u32, Box<dyn Camera>>`** keyed by slot. Each material declares which slot it samples via `Binding::CameraUbo(u32)`. Slot 0 is the conventional default (eagerly created); higher slots get UBOs lazily.
- **Scene-wide pause flag.** `Scene::set_paused(bool)` / `is_paused()` gate physics: while paused the engine skips `fixed_update` dispatch *and clears the fixed-step accumulator* each frame so unpausing doesn't unleash a burst of catch-up ticks. `update` still runs, so an overlay behaviour can poll input and drive the unpause edge. Behaviours that have meaningful `update` work in a paused scene (state machines that consume input, etc.) opt out with an early `if ctx.scene.is_paused() { return; }` — pong's `PhaseController` does this so SPACE can't sneak phase transitions in under the pause overlay.

### `engine::camera`

`Camera` trait (`view()` + `proj(aspect)`) with two implementations:

- **`Camera2D`** — orthographic. `view = translate(-centre)`, `proj` maps `[centre ± half_height*aspect] × [centre ± half_height]` to clip `[-1,1]²`. **Positive Y is downwards** in 2D world space — matches Vulkan clip space without a flip. Exposes `screen_to_world(px, py, extent) -> Vec2` for mouse hit-testing.
- **`Camera3D`** — perspective via `look_at_rh` + `perspective_rh`. **World is Y-up**; the Y-flip lives inside `Camera3D::proj` so on-screen orientation still matches Vulkan clip space.

### `engine::audio`

Thin wrapper over `kira` 0.12 (features `["cpal", "mp3"]`, default-features off). `AudioManager::new()` is infallible — a missing audio device logs once and `play` becomes a silent no-op. `Sound: Clone` is the consumption pattern (the inner kira `Arc` deduplicates samples). Behaviours play via `ctx.audio.play(&sound)`.

### `engine::input`

Polling-only. `is_pressed`, `was_just_pressed`, `was_just_released`, plus mouse equivalents. `KeyCode` and `MouseButton` re-exported from `winit::keyboard::KeyCode` (physical, US-QWERTY position) and `winit::event::MouseButton` so game code never sees winit. Edge state (`was_just_pressed`) stays visible to every dispatch in a frame, cleared once at end-of-frame; `Focused(false)` synthesises releases so Alt-Tabbing during a held key doesn't strand it.

### `engine::time`

Carries both the per-frame variable delta and the 120 Hz fixed step. Phase-aware `delta_time()`. Spiral-of-death cap at 250 ms on the variable delta before it folds into the accumulator. 5-sample running-mean FPS sampler folded in (no separate `FPSLimiter`).

### `engine::app`

`App::new().with_scene(F).run() -> !` is the engine entry point. `App` owns the winit event loop (winit 0.30 `ApplicationHandler`), the `GraphicsManager`, `Resources`, `Scene`, `Time`, `Input`, `AudioManager`. Per-frame loop:

1. `time.begin_frame()` — snapshot variable delta, fold into fixed accumulator.
2. Drain accumulator with `fixed_update` + `apply_commands` per step.
3. `update` + `apply_commands` once.
4. If a behaviour called `ctx.request_scene(builder)`: build the new scene first (shared assets cache-hit against the still-alive old scene's `Rc`s), `mem::replace`, `old.teardown(gm)`, drop old, `apply_commands` on the new scene.
5. `input.end_frame()` + `resources.flush_pending(&mut gm)`.
6. For each populated camera slot: `gm.set_camera(slot, &**camera)`.
7. `Scene::collect_transforms()` → `GraphicsManager::draw_frame(&transforms)`.

Exit is `UpdateCtx::request_exit()` (sets a `&mut bool` the App checks each event). `WindowEvent::CloseRequested` calls `event_loop.exit()` directly.

## Load-bearing decisions

These are the choices that shaped the engine's surface and would be expensive to reverse. The full per-phase decision history was condensed away once it stopped serving the project; what remains is what newcomers (and future-me) need to understand the *why*.

### Architecture & ownership

- **Workspace split (engine + game crates), not a single binary.** Makes the engine reusable as a published library later; keeps game/engine dependency trees separable. The test-3d crate is the worked smoke test that exercises the engine's 3D path without polluting Pong's surface.
- **Trait-based behaviours, not ECS, not embedded scripting.** Matches the project's "small and personal" scope. ECS adds machinery Pong-shaped games don't need; scripting adds runtime overhead and integration complexity for solo work.
- **Engine owns the game loop.** `App::new().with_scene(F).run()` is four lines in `main.rs`. Behaviours are the unit of game logic; this is the only shape that lets that work without exposing winit/Vulkan to game code.
- **Flat scene, no scene graph.** Parent-child transforms are not needed by Pong or most simple 2D games; can be added later as an `Option<parent>` on `Transform` without breaking the API.
- **Resources is a thin RAII layer over the renderer, not a container.** Making `Resources` own `GraphicsManager` would force `Rc<RefCell<GraphicsManager>>` everywhere. Keeping the registries on `GraphicsManager` and passing `&mut GraphicsManager` per loader call matches `Scene`'s "pass the renderer in on demand" pattern.

### Coordinate & lifecycle conventions

- **2D world is Y-down, 3D world is Y-up; both produce Y-down Vulkan clip space.** `Camera2D::proj` is identity-on-Y (Y-down world already matches Vulkan); `Camera3D::proj` negates the Y axis after `perspective_rh` so a Y-up world still renders the right way up. Don't mix conventions inside one scene without being explicit about which transforms live in which frame.
- **RAII via deferred destruction queue, drained at frame boundaries.** Freeing GPU resources needs `device_wait_idle` + renderer-owned Vulkan handles, neither of which `Drop` can do safely from an arbitrary call site (e.g. mid-update). Pushing the id to a `PendingDestroys` queue keeps `Drop` cheap and infallible; `flush_pending` runs once per frame at a known-safe boundary.
- **`apply_commands` runs after every dispatch (every fixed-update substep + the variable update).** Keeps the rule "spawns land at the next dispatch boundary" honest across both hooks. A behaviour spawning from `fixed_update` sees the new object in the next substep of the same frame, not partway through.
- **Edge inputs visible to every dispatch in a frame; cleared once at end-of-frame.** Simplest contract. Lines up with the "one frame = one input snapshot" mental model. Behaviours that mustn't re-fire across substeps transition state in `update` or set a behaviour-local "consumed" flag.

### Renderer & material design

- **Material registry over hardcoded pipelines.** `MaterialDesc` is plain data, not a trait — no per-material boilerplate, and the same descriptor shape supports future hot-reload variants built from filesystem watch events.
- **First instance attribute is always `Mat4`, asserted at registration.** The engine writes the per-instance model matrix at instance offset 0 every frame; making the matrix location per-material would have cost a HashMap probe per instance per frame.
- **`Binding::Sampler2d` flips the batching strategy.** Sampler-using materials share the engine's unit-quad VBO/IBO and batch by `TextureHandle`; sampler-less materials use per-instance `MeshHandle`s and batch by mesh. Matches the existing two batching modes exactly.
- **Always-on depth attachment + per-material `DepthMode`.** A 3D game wanting a 2D HUD needs both depth-tested and depth-free draws in the same scene. Per-material depth keeps the existing `(material, key) -> draw` shape and makes the 2D-HUD use case a one-flag declaration. The 2D built-ins ship `Disabled` to preserve Pong's pre-3D paint-order layering contract.
- **`ModelMesh` is raw bytes + stride.** Vertex layout is owned by the material's `vertex_attrs`; the mesh is just bytes. Same `register_mesh` path serves 2D quads, 3D `pos+normal+uv` cubes, and anything future materials declare.
- **Each material samples at most one camera slot, asserted at registration.** Every realistic use case (3D + 2D HUD, split-screen, minimap) is "this material draws into one camera's view". Allowing multiple would force per-binding slot tracking through the descriptor write path for no consumer.

### Cross-scene & asset sharing

- **Refcounted shared assets + content-addressed `Weak` cache, not explicit per-scene manifests.** `Mesh`/`Texture` are `Rc`-backed clonable handles. Two `load_texture_png(SAME_BYTES)` calls share GPU memory transparently; destruction queues when the last clone drops. The cache holds `Weak`s so it never keeps assets alive on its own. Manifests would give a clean audit surface but at the cost of two declarations per scene and a separate path for runtime spawns.
- **Scene swap runs new builder *before* tearing down the old.** This keeps the old scene's `Rc`s alive *during* the new builder's calls, so a shared loader upgrades the cached `Weak` and shares the GPU resource rather than the obvious-but-wrong "old drops first → cache `Weak` dies → new uploads fresh" path. Synchronous, so the next frame draws against a fully populated new scene.
- **`UpdateCtx::request_scene` over a return-value transition.** Mirrors the existing `request_exit` pattern; last-write-wins; doesn't force every `update` return type to grow a transition enum.
- **One consistent navigation model: Escape always goes back-up; menu Quit is the only exit.** The previous "Escape exits from anywhere" shape was fine for a single-scene game but becomes a footgun the moment a menu exists. Quit on the menu is the explicit single exit point. In-game Escape now opens a pause overlay rather than transitioning directly to the menu — same "back-up" intent, with an extra Continue/Quit confirmation so the player doesn't lose a match to a mis-tapped key. The overlay's Quit option is what actually swaps back to the main menu.

### Dependencies

- **glam over cgmath**, **kira over rodio**. glam is the de-facto Rust gamedev linear-algebra crate, actively maintained, SIMD-backed. kira has per-channel mixing/effects useful for "music + SFX" split in future games.
- **Engine re-exports glam types; pong has no direct glam dep.** Single source of truth for the version pin (`glam = "0.32"`); matches the `KeyCode` re-export pattern. Two direct deps with separately-managed versions would risk silently diverging matrix layouts.
- **winit 0.30 + `ash-window`.** winit 0.30 exposes `raw-window-handle` 0.6 natively, so per-platform surface creation collapses to `ash_window::create_surface`. macOS-specific deps (`cocoa`, `metal`, `objc`) and Windows-specific deps (`winapi`) are gone. Wayland works without the `WINIT_UNIX_BACKEND=x11` override.
- **Physical `KeyCode`, not logical `Key`.** Game controls are position-based — a Dvorak user pressing the W-position key should drive the left paddle up regardless of layout. When a future game needs text input, it grows a logical-key surface alongside `KeyCode`, not on top of it.

## What stays the same

Worth being explicit about, to avoid scope creep:

- Vulkan via `ash` with `share.rs` as tutorial-shaped scaffolding.
- The "mesh once + instances many" + "texture once + instances many" rendering model.
- The unit-quad textured pipeline with alpha-discard fragment shader.
- Handle-based public API with RAII wrappers.
- Solids-then-textures draw order for the 2D built-ins (load-bearing — they run depth-disabled).
- Coordinate convention: positive Y is downwards in 2D world space; 3D world space is Y-up with the Y-flip applied inside `Camera3D::proj`.
