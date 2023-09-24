mod graphics_manager;

use crate::graphics_manager::GraphicsManager;
use crate::graphics_manager::constants::IS_PAINT_FPS_COUNTER;

use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};

struct PongRust {
    graphics_manager: GraphicsManager,
}

impl PongRust {
    pub fn main_loop(mut self, event_loop: EventLoop<()>) {

        let mut tick_counter = graphics_manager::fps_limiter::FPSLimiter::new();

        event_loop.run(move |event, _, control_flow| {

            match event {
                | Event::WindowEvent { event, .. } => {
                    match event {
                        | WindowEvent::CloseRequested => {
                            self.graphics_manager.device_wait_idle();
                            *control_flow = ControlFlow::Exit
                        },
                        | WindowEvent::KeyboardInput { input, .. } => {
                            match input {
                                | KeyboardInput { virtual_keycode, state, .. } => {
                                    match (virtual_keycode, state) {
                                        | (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                            self.graphics_manager.device_wait_idle();
                                            *control_flow = ControlFlow::Exit
                                        },
                                        | _ => {},
                                    }
                                },
                            }
                        },
                        | _ => {},
                    }
                },
                | Event::MainEventsCleared => {
                    self.graphics_manager.window_request_redraw();
                },
                | Event::RedrawRequested(_window_id) => {
                    let delta_time = tick_counter.delta_time(); // 
                    self.graphics_manager.draw_frame(delta_time);

                    if IS_PAINT_FPS_COUNTER {
                        print!("FPS: {}\r", tick_counter.fps());
                    }

                    tick_counter.tick_frame();
                },
                | Event::LoopDestroyed => {
                    self.graphics_manager.device_wait_idle()
                },
                _ => (),
            }

        })
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let graphics_manager = GraphicsManager::new(&event_loop);
    let pong_rust = PongRust {
        graphics_manager,
    };

    pong_rust.main_loop(event_loop);
}