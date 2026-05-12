//! Engine-owned application loop.
//!
//! [`App::run`] owns the winit event loop, the `GraphicsManager`, the
//! `Resources` queue, the [`Input`] state, and the [`Time`] accumulator. The
//! game crate hands it a scene-builder closure
//! (`FnOnce(&mut Resources, &mut GraphicsManager) -> Scene`) and gets to
//! collapse `main` down to roughly:
//!
//! ```ignore
//! engine::app::App::new().with_scene(build_scene).run();
//! ```
//!
//! Per-frame flow on `RedrawRequested`:
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
//! Winit input events update [`Input`] in place — there is no event-dispatch
//! surface for game code. `WindowEvent::Focused(false)` triggers
//! `Input::lose_focus` to avoid stuck-key syndrome after Alt-Tab.

use winit::event::{ElementState, Event as WEvent, KeyboardInput, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::graphics_manager::constants::IS_PAINT_FPS_COUNTER;
use crate::graphics_manager::GraphicsManager;
use crate::input::Input;
use crate::resources::Resources;
use crate::scene::Scene;
use crate::time::Time;

type SceneBuilder = Box<dyn FnOnce(&mut Resources, &mut GraphicsManager) -> Scene>;

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
    /// is up, before the event loop starts. The closure may spawn objects
    /// (deferred — flushed before the first frame).
    pub fn with_scene<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut Resources, &mut GraphicsManager) -> Scene + 'static,
    {
        self.scene_builder = Some(Box::new(f));
        self
    }

    /// Enter the event loop. Diverges: winit 0.20's `EventLoop::run` is `-> !`.
    pub fn run(self) -> ! {
        let builder = self
            .scene_builder
            .expect("App::run requires .with_scene(...)");

        let event_loop = EventLoop::new();
        let mut graphics_manager = GraphicsManager::new(&event_loop);
        let mut resources = Resources::new();
        let mut scene = builder(&mut resources, &mut graphics_manager);
        // Flush build-time spawns so the first frame sees populated objects.
        scene.apply_commands(&mut graphics_manager);

        let mut time = Time::new();
        let mut input = Input::new();
        let mut exit_requested = false;

        event_loop.run(move |event, _, control_flow| {
            if exit_requested {
                graphics_manager.device_wait_idle();
                *control_flow = ControlFlow::Exit;
                return;
            }
            match event {
                WEvent::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        graphics_manager.device_wait_idle();
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::KeyboardInput { input: kb, .. } => {
                        let KeyboardInput { virtual_keycode, state, .. } = kb;
                        if let Some(kc) = virtual_keycode {
                            match state {
                                ElementState::Pressed => input.on_key_pressed(kc),
                                ElementState::Released => input.on_key_released(kc),
                            }
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        input.on_mouse_moved(position.x as f32, position.y as f32);
                    }
                    WindowEvent::MouseInput { state, button, .. } => match state {
                        ElementState::Pressed => input.on_mouse_pressed(button),
                        ElementState::Released => input.on_mouse_released(button),
                    },
                    WindowEvent::Focused(false) => input.lose_focus(),
                    _ => {}
                },
                WEvent::MainEventsCleared => {
                    graphics_manager.window_request_redraw();
                }
                WEvent::RedrawRequested(_) => {
                    time.begin_frame();

                    // Drain accumulator: deterministic physics ticks.
                    time.set_phase_fixed();
                    while time.consume_fixed_step() {
                        scene.dispatch_fixed_update(
                            &time,
                            &input,
                            &mut resources,
                            &mut graphics_manager,
                            &mut exit_requested,
                        );
                        scene.apply_commands(&mut graphics_manager);
                    }

                    // Variable-rate update: rendering-bound work + edge input.
                    time.set_phase_variable();
                    scene.dispatch_update(
                        &time,
                        &input,
                        &mut resources,
                        &mut graphics_manager,
                        &mut exit_requested,
                    );
                    scene.apply_commands(&mut graphics_manager);

                    // Clear edge state *after* both dispatches so every
                    // fixed_update and the update of one frame see the same
                    // `was_just_pressed` / `was_just_released` set.
                    input.end_frame();

                    // Resource RAII flush must happen between frames — see
                    // resources.rs for why we don't do it in Drop.
                    resources.flush_pending(&mut graphics_manager);
                    graphics_manager.set_camera(&scene.camera);
                    let transforms = scene.collect_transforms();
                    graphics_manager.draw_frame(&transforms);

                    if IS_PAINT_FPS_COUNTER {
                        print!("FPS: {}\r", time.fps());
                    }
                }
                WEvent::LoopDestroyed => graphics_manager.device_wait_idle(),
                _ => (),
            }
        })
    }
}
