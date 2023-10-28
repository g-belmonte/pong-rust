mod ball;
mod camera;
mod graphics_manager;
mod paddle;
mod scene;
mod wall;

use crate::graphics_manager::constants::IS_PAINT_FPS_COUNTER;
use crate::graphics_manager::GraphicsManager;

use scene::Scene;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(PartialEq)]
enum GamePhase {
    Start,
    Playing,
    End,
}

enum Action {
    Quit
}

enum PongRustActions {
    SceneAction(scene::Action),
    SystemAction(Action),
}

struct PongRust {
    graphics_manager: GraphicsManager,
    scene: Scene,
    game_phase: GamePhase,
}

impl PongRust {
    pub fn main_loop(mut self, event_loop: EventLoop<()>) {
        let mut tick_counter = graphics_manager::fps_limiter::FPSLimiter::new();

        event_loop.run(move |event, _, control_flow| match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    self.graphics_manager.device_wait_idle();
                    *control_flow = ControlFlow::Exit
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(action) = self.handle_keyboard_input(input) {
                        match action {
                            PongRustActions::SystemAction(Action::Quit) => {
                                self.graphics_manager.device_wait_idle();
                                *control_flow = ControlFlow::Exit
                            },
                            PongRustActions::SceneAction(action) => {
                                self.scene.handle_action(action);
                            }
                        }
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                self.graphics_manager.window_request_redraw();
            }
            Event::RedrawRequested(_window_id) => {
                let delta_time = tick_counter.delta_time();
                if self.scene.game_over() {
                    self.scene.handle_action(scene::Action::GameOver);
                    self.game_phase = GamePhase::End;
                }
                self.scene.update(delta_time);
                let transforms = self.scene.get_model_transforms();
                self.graphics_manager.draw_frame(transforms);

                if IS_PAINT_FPS_COUNTER {
                    print!("FPS: {}\r", tick_counter.fps());
                }

                tick_counter.tick_frame();
            }
            Event::LoopDestroyed => self.graphics_manager.device_wait_idle(),
            _ => (),
        })
    }

    pub fn handle_keyboard_input(&mut self, input: KeyboardInput) -> Option<PongRustActions> {
        let KeyboardInput {
            virtual_keycode,
            state,
            ..
        } = input;
        match (virtual_keycode, state) {
            (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                Some(PongRustActions::SystemAction(Action::Quit))
            },
            (Some(VirtualKeyCode::Space), ElementState::Pressed) => {
                match self.game_phase {
                    GamePhase::Start => {
                        self.game_phase = GamePhase::Playing;
                        Some(PongRustActions::SceneAction(scene::Action::Kickoff))

                    },
                    GamePhase::Playing => None,
                    GamePhase::End => {
                        self.game_phase = GamePhase::Start;
                        Some(PongRustActions::SceneAction(scene::Action::ResetGame))
                    }

                }
            },
            (Some(VirtualKeyCode::W), ElementState::Pressed) => {
                Some(PongRustActions::SceneAction(scene::Action::LeftPaddleUp))
            },
            (Some(VirtualKeyCode::W), ElementState::Released) => {
                Some(PongRustActions::SceneAction(scene::Action::LeftPaddleStop))
            },
            (Some(VirtualKeyCode::S), ElementState::Pressed) => {
                Some(PongRustActions::SceneAction(scene::Action::LeftPaddleDown))
            },
            (Some(VirtualKeyCode::S), ElementState::Released) => {
                Some(PongRustActions::SceneAction(scene::Action::LeftPaddleStop))
            },
            (Some(VirtualKeyCode::I), ElementState::Pressed) => {
                Some(PongRustActions::SceneAction(scene::Action::RightPaddleUp))
            },
            (Some(VirtualKeyCode::I), ElementState::Released) => {
                Some(PongRustActions::SceneAction(scene::Action::RightPaddleStop))
            },
            (Some(VirtualKeyCode::K), ElementState::Pressed) => {
                Some(PongRustActions::SceneAction(scene::Action::RightPaddleDown))
            },
            (Some(VirtualKeyCode::K), ElementState::Released) => {
                Some(PongRustActions::SceneAction(scene::Action::RightPaddleStop))
            },
            _ => None
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let scene = Scene::new();
    let graphics_manager = GraphicsManager::new(&event_loop, &scene);
    let pong_rust = PongRust {
        graphics_manager,
        scene,
        game_phase: GamePhase::Start,
    };

    pong_rust.main_loop(event_loop);
}
