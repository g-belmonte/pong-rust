//! Engine-owned application loop.
//!
//! [`App::run`] owns the winit event loop, the `GraphicsManager`, the
//! `Resources` queue, the [`Input`] state, the [`Time`] accumulator, and the
//! `AudioManager`. The game crate hands it a scene-builder closure
//! (`FnOnce(&mut Resources, &mut GraphicsManager) -> Scene`) and gets to
//! collapse `main` down to roughly:
//!
//! ```ignore
//! engine::app::App::new().with_scene(build_scene).run();
//! ```
//!
//! Per-frame flow on `WindowEvent::RedrawRequested`:
//!   1. `Time::begin_frame` snapshots the variable delta and folds it into
//!      the fixed-step accumulator.
//!   2. While the accumulator has a step available:
//!        a. `Scene::dispatch_fixed_update(time, input, …)` runs every
//!           behaviour's `fixed_update`.
//!        b. `Scene::apply_commands` flushes any spawn/despawn queued there.
//!   3. `Scene::dispatch_update(time, input, …)` runs every behaviour's
//!      variable-step `update`.
//!   4. `Scene::apply_commands` flushes the variable-update queue.
//!   5. `Input::end_frame` clears `just_pressed` / `just_released` edges.
//!   6. `Resources::flush_pending` drops queued GPU resources (RAII Mesh/Texture).
//!   7. `GraphicsManager::set_camera` pushes the scene's `Camera2D` to the
//!      shared UBO (no-op when matrices unchanged from the previous frame).
//!   8. `Scene::collect_transforms` gathers `(handle, matrix)` for the renderer.
//!   9. `GraphicsManager::draw_frame` submits.
//!
//! ## winit 0.30 integration
//!
//! winit 0.30 replaced the `EventLoop::run(closure)` shape with the
//! [`ApplicationHandler`] trait. The renderer + scene cannot be built until
//! the platform delivers a `resumed` event (on Android/iOS this is when the
//! surface becomes available; on desktop it fires once at startup). So the
//! engine state lives behind `Option`s on [`AppState`] and is initialised the
//! first time `resumed` fires.
//!
//! winit 0.30's `run_app` returns `Result<(), EventLoopError>` (no longer
//! `-> !`), but we keep `App::run() -> !` and call `process::exit` after the
//! loop returns. Reason: avoids reshaping the documented destructor
//! workarounds (end-of-init pipeline-cache save; per-frame
//! `Resources::flush_pending`) in the same PR as the winit bump.
//!
//! Winit input events update [`Input`] in place — there is no event-dispatch
//! surface for game code. `WindowEvent::Focused(false)` triggers
//! `Input::lose_focus` to avoid stuck-key syndrome after Alt-Tab.

use std::process;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::WindowId;

use crate::audio::AudioManager;
use crate::graphics_manager::constants::IS_PAINT_FPS_COUNTER;
use crate::graphics_manager::GraphicsManager;
use crate::input::Input;
use crate::resources::Resources;
use crate::scene::{Scene, SceneBuilder};
use crate::time::Time;

pub struct App {
    scene_builder: Option<SceneBuilder>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        Self { scene_builder: None }
    }

    /// Register the scene-construction closure. Called once after the renderer
    /// is up, before the event loop starts driving frames. The closure may
    /// spawn objects (deferred — flushed before the first frame).
    pub fn with_scene<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut Resources, &mut GraphicsManager) -> Scene + 'static,
    {
        self.scene_builder = Some(Box::new(f));
        self
    }

    /// Enter the event loop. Diverges via `process::exit` after winit returns.
    pub fn run(self) -> ! {
        let builder = self
            .scene_builder
            .expect("App::run requires .with_scene(...)");

        let event_loop = EventLoop::new().expect("Failed to create event loop");
        // Poll so the loop runs at the renderer's pace, not at OS event arrival.
        event_loop.set_control_flow(ControlFlow::Poll);

        let mut state = AppState {
            scene_builder: Some(builder),
            engine: None,
            exit_requested: false,
        };

        if let Err(e) = event_loop.run_app(&mut state) {
            eprintln!("event loop error: {e}");
            process::exit(1);
        }
        process::exit(0);
    }
}

/// Initialised engine state. Constructed on the first `resumed` event.
struct Engine {
    graphics_manager: GraphicsManager,
    resources: Resources,
    scene: Scene,
    time: Time,
    input: Input,
    audio: AudioManager,
}

struct AppState {
    scene_builder: Option<SceneBuilder>,
    engine: Option<Engine>,
    exit_requested: bool,
}

impl ApplicationHandler for AppState {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.engine.is_some() {
            // Re-resume (e.g. mobile foregrounding). Desktop platforms only
            // fire this once; no state to rebuild here today.
            return;
        }
        let builder = self
            .scene_builder
            .take()
            .expect("scene builder consumed before first resume");

        let mut graphics_manager = GraphicsManager::new(event_loop);
        let mut resources = Resources::new();
        let mut scene = builder(&mut resources, &mut graphics_manager);
        // Flush build-time spawns so the first frame sees populated objects.
        scene.apply_commands(&mut graphics_manager);

        self.engine = Some(Engine {
            graphics_manager,
            resources,
            scene,
            time: Time::new(),
            input: Input::new(),
            audio: AudioManager::new(),
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(engine) = self.engine.as_mut() else {
            return;
        };

        if self.exit_requested {
            engine.graphics_manager.device_wait_idle();
            event_loop.exit();
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                engine.graphics_manager.device_wait_idle();
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event: key_event, .. } => {
                // Engine input maps to *physical* keys so layouts (AZERTY,
                // Dvorak) don't shift WASD/IJKL. `PhysicalKey::Unidentified`
                // is dropped — no game cares about unknown scancodes.
                if let PhysicalKey::Code(code) = key_event.physical_key {
                    // winit 0.30 surfaces an explicit `repeat` flag; the
                    // engine's `Input` already filters repeats by tracking
                    // pressed state, but the explicit short-circuit avoids
                    // the HashSet roundtrip.
                    if key_event.repeat {
                        return;
                    }
                    match key_event.state {
                        ElementState::Pressed => engine.input.on_key_pressed(code),
                        ElementState::Released => engine.input.on_key_released(code),
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                engine
                    .input
                    .on_mouse_moved(position.x as f32, position.y as f32);
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => engine.input.on_mouse_pressed(button),
                ElementState::Released => engine.input.on_mouse_released(button),
            },
            WindowEvent::Focused(false) => engine.input.lose_focus(),
            WindowEvent::RedrawRequested => {
                redraw(engine, &mut self.exit_requested);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        // winit 0.30 replaced `MainEventsCleared` with this hook. Request a
        // redraw each turn through the loop to drive the render at vsync /
        // mailbox cadence.
        if let Some(engine) = self.engine.as_mut() {
            engine.graphics_manager.window_request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(engine) = self.engine.as_mut() {
            engine.graphics_manager.device_wait_idle();
        }
    }
}

fn redraw(engine: &mut Engine, exit_requested: &mut bool) {
    engine.time.begin_frame();

    let mut pending_scene: Option<SceneBuilder> = None;

    // Drain accumulator: deterministic physics ticks.
    engine.time.set_phase_fixed();
    while engine.time.consume_fixed_step() {
        engine.scene.dispatch_fixed_update(
            &engine.time,
            &engine.input,
            &mut engine.resources,
            &mut engine.graphics_manager,
            &mut engine.audio,
            exit_requested,
            &mut pending_scene,
        );
        engine.scene.apply_commands(&mut engine.graphics_manager);
    }

    // Variable-rate update: rendering-bound work + edge input.
    engine.time.set_phase_variable();
    engine.scene.dispatch_update(
        &engine.time,
        &engine.input,
        &mut engine.resources,
        &mut engine.graphics_manager,
        &mut engine.audio,
        exit_requested,
        &mut pending_scene,
    );
    engine.scene.apply_commands(&mut engine.graphics_manager);

    // Scene swap. Order matters for asset reuse: build the new scene *first*
    // (its loader calls cache-hit against the still-alive old scene's Rcs,
    // bumping refcount), then tear down the old scene (refcount falls; only
    // assets not also held by the new scene hit zero and queue), then
    // flush_pending (destroys those). Synchronous, so the next frame draws
    // against a fully-populated new scene; the *current* frame still draws
    // the (now torn-down) old scene one last time — see below.
    if let Some(builder) = pending_scene.take() {
        let new_scene = builder(&mut engine.resources, &mut engine.graphics_manager);
        let mut old = std::mem::replace(&mut engine.scene, new_scene);
        old.teardown(&mut engine.graphics_manager);
        drop(old);
        engine.scene.apply_commands(&mut engine.graphics_manager);
    }

    // Clear edge state *after* both dispatches so every fixed_update and the
    // update of one frame see the same `was_just_pressed` / `was_just_released`
    // set.
    engine.input.end_frame();

    // Apply any file-watcher-driven asset reloads (textures, meshes, sounds)
    // before we flush pending destroys and draw. Compiles to a no-op without
    // the `hot-reload` feature.
    engine
        .resources
        .process_hot_reloads(&mut engine.graphics_manager);
    // Resource RAII flush must happen between frames — see resources.rs for
    // why we don't do it in Drop. Also reclaims assets dropped during the
    // scene swap above.
    engine.resources.flush_pending(&mut engine.graphics_manager);
    // Push every populated camera slot. Cache-checked per slot, so static
    // cameras only pay the actual `device_wait_idle` + UBO-write cost the
    // first time they change.
    for (&slot, camera) in engine.scene.cameras.iter() {
        engine.graphics_manager.set_camera(slot, &**camera);
    }
    let transforms = engine.scene.collect_transforms();
    engine.graphics_manager.draw_frame(&transforms);

    if IS_PAINT_FPS_COUNTER {
        print!("FPS: {}\r", engine.time.fps());
    }
}
