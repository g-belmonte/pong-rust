//! Pong's scene-building function.
//!
//! Loads shared resources (meshes, textures, font atlas) and spawns every
//! game object into the engine's `Scene`. All game state then lives in
//! behaviours — paddles, ball, walls, digits, labels, and a `PhaseController`
//! that owns the game-state machine and label visibility.
//!
//! ## Coordinate convention
//!
//! **Positive Y is downwards.** Walls sit at `y = ±WALL_OFFSET_Y`, paddles
//! move with negative-y = up, digits sit above the play field at `y ≈ -3.7`.
//! Matches Vulkan clip-space Y and is mirrored by `digit.rs` and `text.rs`.

use engine::{Vec2, Vec3};

use engine::camera::Camera2D;
use engine::graphics_manager::structures::rect_mesh;
use engine::graphics_manager::GraphicsManager;
use engine::resources::Resources;
use engine::input::KeyCode;
use engine::scene::{Object, Renderable, Scene, Transform};

use crate::ball::BallBehaviour;
use crate::digit::{DigitBehaviour, DigitMeshes};
use crate::paddle::PaddleBehaviour;
use crate::phase::PhaseController;
use crate::text::{FontAtlas, TextLabelBehaviour};
use crate::wall::WallBehaviour;

const PADDLE_HEIGHT: f32 = 2.0;
const PADDLE_WIDTH: f32 = 0.2;
const WALL_HEIGHT: f32 = 0.2;
const WALL_WIDTH: f32 = 10.0;
const DIGIT_SEGMENT_SIZE: f32 = 0.4;
const BALL_SIDE_LENGTH: f32 = 0.2;

// Walls placed symmetrically around y = 0; top at -WALL_OFFSET_Y, bottom at +.
const WALL_OFFSET_Y: f32 = 3.2;
// Ball's |x| past this counts as a goal — slightly outside paddle x (±4.0).
const GOAL_LINE_X: f32 = 4.7;
const PADDLE_SPEED: f32 = 2.0;
const BALL_KICKOFF_SPEED_X: f32 = 4.0;

// World-space half-height of the visible area. Chosen to preserve the
// pre-Phase-4 perspective view: at the previous camera (z=10, fov=45°),
// the visible height at z=0 was 2 * 10 * tan(22.5°) ≈ 8.284, so half_height
// ≈ 4.142 keeps every existing position (walls at ±3.2, digits at y ≈ -3.7)
// in the same on-screen spot.
const CAMERA_HALF_HEIGHT: f32 = 4.142;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");
const FONT_RASTER_PX: f32 = 48.0;
// 48 px / 0.005 = 9600 px per world unit — sized so "Welcome" sits inside
// the play field. Kept in sync with FONT_RASTER_PX.
const FONT_WORLD_SCALE: f32 = 0.005;

// All four events currently share the same source file. When distinct samples
// land, give each its own `include_bytes!` line and the rest of the wiring
// stays the same.
const WALL_BOUNCE_AUDIO: &[u8] = include_bytes!("../assets/hit.mp3");
const PADDLE_BOUNCE_AUDIO: &[u8] = include_bytes!("../assets/hit.mp3");
const SCORE_AUDIO: &[u8] = include_bytes!("../assets/score.mp3");
const GAME_OVER_AUDIO: &[u8] = include_bytes!("../assets/victory.mp3");

pub const WINNING_SCORE: u8 = 9;

mod color {
    pub const RED: [f32; 3] = [1.0, 0.0, 0.0];
    pub const BLUE: [f32; 3] = [0.0, 0.0, 1.0];
    pub const GREEN: [f32; 3] = [0.0, 1.0, 0.0];
}

/// Build the initial scene. Handed to `engine::app::App::with_scene`.
pub fn build_scene(resources: &mut Resources, gm: &mut GraphicsManager) -> Scene {
    let mut scene = Scene::new();
    scene.set_camera(0, Box::new(Camera2D::new(Vec2::ZERO, CAMERA_HALF_HEIGHT)));

    // Shared resources. Their RAII wrappers move into PhaseController below
    // so they outlive every instance built against them.
    let paddle_mesh = resources.load_mesh(gm, &rect_mesh(PADDLE_WIDTH, PADDLE_HEIGHT));
    let wall_mesh = resources.load_mesh(gm, &rect_mesh(WALL_WIDTH, WALL_HEIGHT));
    let digit_meshes = DigitMeshes::load(resources, gm, DIGIT_SEGMENT_SIZE);
    let font_atlas = FontAtlas::build(resources, gm, FONT_BYTES, FONT_RASTER_PX);

    // Sound assets: cheap to clone (kira's StaticSoundData is Arc-backed).
    // Ball owns the bounce clones; PhaseController owns the score/end clones.
    let wall_bounce_sfx = resources.load_sound(WALL_BOUNCE_AUDIO);
    let paddle_bounce_sfx = resources.load_sound(PADDLE_BOUNCE_AUDIO);
    let score_sfx = resources.load_sound(SCORE_AUDIO);
    let game_over_sfx = resources.load_sound(GAME_OVER_AUDIO);

    // Walls first so paddles/ball can capture their IDs.
    let top_wall = scene.spawn(
        Object::new()
            .with_position(Vec3::new(0.0, -WALL_OFFSET_Y, 0.0))
            .with_renderable(Renderable::Solid {
                mesh: wall_mesh.handle(),
                color: color::GREEN,
            })
            .with_behaviour(WallBehaviour::new(WALL_HEIGHT)),
    );
    let bottom_wall = scene.spawn(
        Object::new()
            .with_position(Vec3::new(0.0, WALL_OFFSET_Y, 0.0))
            .with_renderable(Renderable::Solid {
                mesh: wall_mesh.handle(),
                color: color::GREEN,
            })
            .with_behaviour(WallBehaviour::new(WALL_HEIGHT)),
    );

    let left_paddle = scene.spawn(
        Object::new()
            .with_position(Vec3::new(-4.0, 0.0, 0.0))
            .with_renderable(Renderable::Solid {
                mesh: paddle_mesh.handle(),
                color: color::RED,
            })
            .with_behaviour(PaddleBehaviour::new(
                PADDLE_WIDTH,
                PADDLE_HEIGHT,
                PADDLE_SPEED,
                KeyCode::KeyW,
                KeyCode::KeyS,
                top_wall,
                bottom_wall,
            )),
    );
    let right_paddle = scene.spawn(
        Object::new()
            .with_position(Vec3::new(4.0, 0.0, 0.0))
            .with_renderable(Renderable::Solid {
                mesh: paddle_mesh.handle(),
                color: color::BLUE,
            })
            .with_behaviour(PaddleBehaviour::new(
                PADDLE_WIDTH,
                PADDLE_HEIGHT,
                PADDLE_SPEED,
                KeyCode::KeyI,
                KeyCode::KeyK,
                top_wall,
                bottom_wall,
            )),
    );

    let (ball_behaviour, ball_renderable) = BallBehaviour::load(
        resources,
        gm,
        BALL_SIDE_LENGTH,
        BALL_KICKOFF_SPEED_X,
        top_wall,
        bottom_wall,
        left_paddle,
        right_paddle,
        wall_bounce_sfx,
        paddle_bounce_sfx,
    );
    let ball = scene.spawn(
        Object::new()
            .with_transform(Transform {
                position: Vec3::ZERO,
                scale: Vec3::new(BALL_SIDE_LENGTH, BALL_SIDE_LENGTH, 1.0),
                ..Transform::default()
            })
            .with_renderable(ball_renderable)
            .with_behaviour(ball_behaviour),
    );

    // Digits sit above the play field (y < 0 in this Y-down convention).
    let left_digit = scene.spawn(
        Object::new()
            .with_position(Vec3::new(-1.0, -3.7, 0.0))
            .with_behaviour(DigitBehaviour::new(gm, &digit_meshes)),
    );
    let right_digit = scene.spawn(
        Object::new()
            .with_position(Vec3::new(1.0, -3.7, 0.0))
            .with_behaviour(DigitBehaviour::new(gm, &digit_meshes)),
    );

    let welcome_label = scene.spawn(
        Object::new()
            .with_position(Vec3::new(0.0, -1.2, 0.0))
            .with_behaviour(TextLabelBehaviour::new(
                gm,
                &font_atlas,
                "Welcome",
                FONT_WORLD_SCALE,
                true,
            )),
    );
    let game_over_label = scene.spawn(
        Object::new()
            .with_position(Vec3::new(0.0, -1.2, 0.0))
            .with_behaviour(TextLabelBehaviour::new(
                gm,
                &font_atlas,
                "Game Over",
                FONT_WORLD_SCALE,
                false,
            )),
    );

    scene.spawn(
        Object::new().with_behaviour(PhaseController::new(
            ball,
            left_paddle,
            right_paddle,
            left_digit,
            right_digit,
            welcome_label,
            game_over_label,
            WINNING_SCORE,
            GOAL_LINE_X,
            paddle_mesh,
            wall_mesh,
            digit_meshes,
            font_atlas,
            score_sfx,
            game_over_sfx,
        )),
    );

    scene
}
