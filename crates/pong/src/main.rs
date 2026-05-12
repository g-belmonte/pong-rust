mod ball;
mod digit;
mod paddle;
mod scene;
mod text;
mod wall;

use engine::graphics_manager;
use engine::graphics_manager::constants::IS_PAINT_FPS_COUNTER;
use engine::graphics_manager::GraphicsManager;

use scene::Scene;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(PartialEq)]
enum GamePhase {
    Start,
    Playing,
    End,
}

enum SystemAction {
    Quit,
}

/// Outcome of a single keyboard event: either a game-scene action or a
/// process-level action like quitting.
enum InputAction {
    Scene(scene::Action),
    System(SystemAction),
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
                            InputAction::System(SystemAction::Quit) => {
                                self.graphics_manager.device_wait_idle();
                                *control_flow = ControlFlow::Exit
                            }
                            InputAction::Scene(action) => {
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
                if self.game_phase == GamePhase::Playing && self.scene.game_over() {
                    self.scene.handle_action(scene::Action::GameOver);
                    if self.scene.match_over() {
                        self.game_phase = GamePhase::End;
                        self.scene.set_game_over_visible(true);
                    } else {
                        self.scene.handle_action(scene::Action::ResetRound);
                        self.game_phase = GamePhase::Start;
                        self.scene.set_welcome_visible(true);
                    }
                }
                self.scene.update(delta_time);
                let transforms = self.scene.get_model_transforms();
                self.graphics_manager.draw_frame(&transforms);

                if IS_PAINT_FPS_COUNTER {
                    print!("FPS: {}\r", tick_counter.fps());
                }

                tick_counter.tick_frame();
            }
            Event::LoopDestroyed => self.graphics_manager.device_wait_idle(),
            _ => (),
        })
    }

    pub fn handle_keyboard_input(&mut self, input: KeyboardInput) -> Option<InputAction> {
        let KeyboardInput {
            virtual_keycode,
            state,
            ..
        } = input;
        let act = |a| Some(InputAction::Scene(a));
        match (virtual_keycode, state) {
            (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                Some(InputAction::System(SystemAction::Quit))
            }
            (Some(VirtualKeyCode::Space), ElementState::Pressed) => match self.game_phase {
                GamePhase::Start => {
                    self.game_phase = GamePhase::Playing;
                    self.scene.set_welcome_visible(false);
                    act(scene::Action::Kickoff)
                }
                GamePhase::Playing => None,
                GamePhase::End => {
                    self.game_phase = GamePhase::Start;
                    self.scene.set_game_over_visible(false);
                    self.scene.set_welcome_visible(true);
                    act(scene::Action::ResetGame)
                }
            },
            (Some(VirtualKeyCode::W), ElementState::Pressed) => act(scene::Action::LeftPaddleUp),
            (Some(VirtualKeyCode::W), ElementState::Released) => act(scene::Action::LeftPaddleStop),
            (Some(VirtualKeyCode::S), ElementState::Pressed) => act(scene::Action::LeftPaddleDown),
            (Some(VirtualKeyCode::S), ElementState::Released) => act(scene::Action::LeftPaddleStop),
            (Some(VirtualKeyCode::I), ElementState::Pressed) => act(scene::Action::RightPaddleUp),
            (Some(VirtualKeyCode::I), ElementState::Released) => act(scene::Action::RightPaddleStop),
            (Some(VirtualKeyCode::K), ElementState::Pressed) => act(scene::Action::RightPaddleDown),
            (Some(VirtualKeyCode::K), ElementState::Released) => act(scene::Action::RightPaddleStop),
            _ => None,
        }
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let mut graphics_manager = GraphicsManager::new(&event_loop);
    let scene = Scene::new(&mut graphics_manager);
    let pong_rust = PongRust {
        graphics_manager,
        scene,
        game_phase: GamePhase::Start,
    };

    pong_rust.main_loop(event_loop);
}
