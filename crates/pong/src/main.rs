mod ball;
mod digit;
mod paddle;
mod phase;
mod scene_game;
mod scene_menu;
mod text;
mod tinted_text;
mod wall;

fn main() {
    engine::app::App::new()
        .with_scene(scene_menu::build_menu)
        .run();
}
