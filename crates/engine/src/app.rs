//! Engine-owned application loop.
//!
//! [`App::run`] owns the winit event loop, the `GraphicsManager`, and the
//! `Resources` queue; the game crate hands it a scene-builder closure
//! (`FnOnce(&mut Resources, &mut GraphicsManager) -> Scene`) and gets to
//! collapse `main` down to roughly:
//!
//! ```ignore
//! engine::app::App::new().with_scene(build_scene).run();
//! ```
//!
//! Per-frame flow on `RedrawRequested`:
//!   1. `Scene::dispatch_update(dt)` runs every behaviour's `update` hook.
//!   2. `Scene::apply_commands` flushes any spawn/despawn queued during update.
//!   3. `Resources::flush_pending` drops queued GPU resources (RAII Mesh/Texture).
//!   4. `Scene::collect_transforms` gathers `(handle, matrix)` for the renderer.
//!   5. `GraphicsManager::draw_frame` submits.
//!
//! Winit keyboard events are translated to [`Event::KeyPressed`] /
//! [`Event::KeyReleased`] and dispatched via `Scene::dispatch_event`.
//! Phase 3 will replace that with a polling `Input` abstraction.

use winit::event::{ElementState, Event as WEvent, KeyboardInput, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use crate::graphics_manager::constants::IS_PAINT_FPS_COUNTER;
use crate::graphics_manager::fps_limiter::FPSLimiter;
use crate::graphics_manager::GraphicsManager;
use crate::resources::Resources;
use crate::scene::{Event, Scene};

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

        let mut tick_counter = FPSLimiter::new();
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
                    WindowEvent::KeyboardInput { input, .. } => {
                        let KeyboardInput { virtual_keycode, state, .. } = input;
                        if let Some(kc) = virtual_keycode {
                            let engine_event = match state {
                                ElementState::Pressed => Event::KeyPressed(kc),
                                ElementState::Released => Event::KeyReleased(kc),
                            };
                            scene.dispatch_event(
                                &engine_event,
                                &mut resources,
                                &mut graphics_manager,
                                &mut exit_requested,
                            );
                            scene.apply_commands(&mut graphics_manager);
                        }
                    }
                    _ => {}
                },
                WEvent::MainEventsCleared => {
                    graphics_manager.window_request_redraw();
                }
                WEvent::RedrawRequested(_) => {
                    let dt = tick_counter.delta_time();
                    scene.dispatch_update(
                        dt,
                        &mut resources,
                        &mut graphics_manager,
                        &mut exit_requested,
                    );
                    scene.apply_commands(&mut graphics_manager);
                    // Resource RAII flush must happen between frames — see
                    // resources.rs for why we don't do it in Drop.
                    resources.flush_pending(&mut graphics_manager);
                    let transforms = scene.collect_transforms();
                    graphics_manager.draw_frame(&transforms);

                    if IS_PAINT_FPS_COUNTER {
                        print!("FPS: {}\r", tick_counter.fps());
                    }
                    tick_counter.tick_frame();
                }
                WEvent::LoopDestroyed => graphics_manager.device_wait_idle(),
                _ => (),
            }
        })
    }
}
