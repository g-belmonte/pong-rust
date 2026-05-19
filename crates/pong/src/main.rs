mod ball;
mod digit;
mod paddle;
mod pause;
mod phase;
mod scene_game;
mod scene_menu;
mod scene_settings;
mod settings;
mod text;
mod tinted_text;
mod wall;

use std::cell::RefCell;
use std::rc::Rc;

use settings::Settings;

fn main() {
    // Load persisted settings (or fall back to defaults). The Rc<RefCell<_>>
    // is cloned into every scene-builder closure that needs it; the game
    // scene reads values at construction, the settings scene mutates them
    // and saves to disk on Escape.
    let settings = Rc::new(RefCell::new(Settings::load()));
    let initial = Rc::clone(&settings);
    engine::app::App::new()
        .with_scene(move |res, gm| scene_menu::build_menu(res, gm, initial))
        .run();
}
