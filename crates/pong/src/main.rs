mod ball;
mod digit;
mod paddle;
mod phase;
mod scene;
mod text;
mod wall;

fn main() {
    engine::app::App::new()
        .with_scene(scene::build_scene)
        .run();
}
