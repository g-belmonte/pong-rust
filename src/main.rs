mod camera;
mod graphics_manager;
mod paddle;
mod scene;

use crate::graphics_manager::constants::IS_PAINT_FPS_COUNTER;
use crate::graphics_manager::GraphicsManager;

use scene::Scene;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

struct PongRust {
    graphics_manager: GraphicsManager,
    scene: Scene,
}

impl PongRust {
    pub fn main_loop(mut self, event_loop: EventLoop<()>) {
        let mut tick_counter = graphics_manager::fps_limiter::FPSLimiter::new();

        event_loop.run(move |event, _, control_flow| {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => {
                        self.graphics_manager.device_wait_idle();
                        *control_flow = ControlFlow::Exit
                    }
                    WindowEvent::KeyboardInput { input, .. } => match input {
                        KeyboardInput {
                            virtual_keycode,
                            state,
                            ..
                        } => match (virtual_keycode, state) {
                            (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                self.graphics_manager.device_wait_idle();
                                *control_flow = ControlFlow::Exit
                            }
                            _ => {}
                        },
                    },
                    _ => {}
                },
                Event::MainEventsCleared => {
                    self.graphics_manager.window_request_redraw();
                }
                Event::RedrawRequested(_window_id) => {
                    let _delta_time = tick_counter.delta_time();
                    // TODO: simulate changes using the delta time
                    let transforms = self.scene.get_model_transforms();
                    self.graphics_manager.draw_frame(transforms);

                    if IS_PAINT_FPS_COUNTER {
                        print!("FPS: {}\r", tick_counter.fps());
                    }

                    tick_counter.tick_frame();
                }
                Event::LoopDestroyed => self.graphics_manager.device_wait_idle(),
                _ => (),
            }
        })
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let scene = Scene::new();
    let graphics_manager = GraphicsManager::new(&event_loop, &scene);
    let pong_rust = PongRust { graphics_manager, scene };

    pong_rust.main_loop(event_loop);
}
