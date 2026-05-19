//! Settings scene.
//!
//! "Settings" big title at top, three editable rows in the middle (winning
//! score, ball speed, paddle speed) and a small "press ESC to return"
//! hint right-aligned in the bottom corner of the typical-aspect visible
//! area.
//!
//! Navigation: Up/Down moves between rows; Left/Right adjusts the selected
//! row by one step (see `crate::settings` for ranges + step sizes); Escape
//! saves to disk and `request_scene`s back to the main menu. Changes take
//! effect at the next `scene_game::build_game` call — a match already in
//! progress keeps its baked-in values.
//!
//! The value display ("< 9 >") is a `TintedTextLabelBehaviour` that gets
//! rebuilt via `set_text` on every adjustment; the static row labels
//! ("Winning score:") never change. Selected row is shown white at scale
//! 1.25; others are dim grey at scale 1.0 (same visual language as the
//! main menu).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use engine::camera::Camera2D;
use engine::graphics_manager::{GraphicsManager, MaterialHandle};
use engine::input::KeyCode;
use engine::resources::{FontAtlas, Resources};
use engine::scene::{Behaviour, Object, ObjectId, Scene, Transform, UpdateCtx};
use engine::{Vec2, Vec3};

use crate::scene_menu::register_tinted_text_material;
use crate::settings::Settings;
use crate::text::TextLabelBehaviour;
use crate::tinted_text::TintedTextLabelBehaviour;

const FONT_RASTER_PX: f32 = 48.0;
const FONT_WORLD_SCALE: f32 = 0.005;

const TITLE_SCALE: f32 = 2.0;
const ROW_SCALE_NORMAL: f32 = 1.0;
const ROW_SCALE_SELECTED: f32 = 1.25;
const COLOR_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const COLOR_GREY: [f32; 3] = [0.45, 0.45, 0.45];

const CAMERA_HALF_HEIGHT: f32 = 4.142;

const TITLE_Y: f32 = -3.0;
const ROW_NAMES: [&str; 3] = ["Winning score:", "Ball speed:", "Paddle speed:"];
const ROW_Y: [f32; 3] = [-0.9, 0.0, 0.9];
/// X column where the row labels are centred (slightly left of origin so the
/// value column sits on the right).
const ROW_LABEL_X: f32 = -1.5;
/// X column for the editable value ("< 9 >").
const ROW_VALUE_X: f32 = 1.2;

/// Bottom-right hint position. Chosen for typical landscape aspect (4:3 to
/// 16:9, half_height = 4.142 → visible x ∈ [-5.5, -7.4]). On very narrow
/// windows the label can clip; a future per-frame "track the bottom-right
/// corner" behaviour is the cleanup path.
const HINT_POS: Vec3 = Vec3::new(3.4, 3.8, 0.0);

pub fn build_settings(
    resources: &mut Resources,
    gm: &mut GraphicsManager,
    settings: Rc<RefCell<Settings>>,
) -> Scene {
    let mut scene = Scene::new();
    scene.set_camera(0, Box::new(Camera2D::new(Vec2::ZERO, CAMERA_HALF_HEIGHT)));

    let atlas = resources.load_font(gm, engine::asset!("assets/DejaVuSans.ttf"), FONT_RASTER_PX);
    let tinted_material = register_tinted_text_material(resources, gm);

    // Title
    scene.spawn(
        Object::new()
            .with_transform(Transform {
                position: Vec3::new(0.0, TITLE_Y, 0.0),
                scale: Vec3::splat(TITLE_SCALE),
                ..Transform::default()
            })
            .with_behaviour(TintedTextLabelBehaviour::new(
                gm,
                &atlas,
                tinted_material,
                "Settings",
                FONT_WORLD_SCALE,
                COLOR_WHITE,
                true,
            )),
    );

    // Static row labels (left column). Spawned with the row's *selected*
    // colour iff i == 0 so the first frame looks right; the controller
    // mutates them every tick anyway.
    let mut label_ids: Vec<ObjectId> = Vec::new();
    for (i, &name) in ROW_NAMES.iter().enumerate() {
        let (color, scale) = if i == 0 {
            (COLOR_WHITE, ROW_SCALE_SELECTED)
        } else {
            (COLOR_GREY, ROW_SCALE_NORMAL)
        };
        let id = scene.spawn(
            Object::new()
                .with_transform(Transform {
                    position: Vec3::new(ROW_LABEL_X, ROW_Y[i], 0.0),
                    scale: Vec3::splat(scale),
                    ..Transform::default()
                })
                .with_behaviour(TintedTextLabelBehaviour::new(
                    gm,
                    &atlas,
                    tinted_material,
                    name,
                    FONT_WORLD_SCALE,
                    color,
                    true,
                )),
        );
        label_ids.push(id);
    }

    // Value labels (right column). Each rebuilds its text on adjustment.
    let mut value_ids: Vec<ObjectId> = Vec::new();
    for (i, &y) in ROW_Y.iter().enumerate() {
        let (color, scale) = if i == 0 {
            (COLOR_WHITE, ROW_SCALE_SELECTED)
        } else {
            (COLOR_GREY, ROW_SCALE_NORMAL)
        };
        let text = format_value(i, &settings.borrow());
        let id = scene.spawn(
            Object::new()
                .with_transform(Transform {
                    position: Vec3::new(ROW_VALUE_X, y, 0.0),
                    scale: Vec3::splat(scale),
                    ..Transform::default()
                })
                .with_behaviour(TintedTextLabelBehaviour::new(
                    gm,
                    &atlas,
                    tinted_material,
                    &text,
                    FONT_WORLD_SCALE,
                    color,
                    true,
                )),
        );
        value_ids.push(id);
    }

    // Bottom-right hint. Regular (untinted) TextLabel — always white.
    scene.spawn(
        Object::new()
            .with_position(HINT_POS)
            .with_behaviour(TextLabelBehaviour::new(
                gm,
                &atlas,
                "press ESC to return",
                FONT_WORLD_SCALE,
                true,
            )),
    );

    scene.spawn(Object::new().with_behaviour(SettingsController {
        selected_row: 0,
        label_ids,
        value_ids,
        settings,
        tinted_material,
        atlas,
    }));

    scene
}

fn format_value(row: usize, settings: &Settings) -> String {
    match row {
        0 => format!("< {} >", settings.winning_score),
        1 => format!("< {:.1} >", settings.ball_speed),
        2 => format!("< {:.1} >", settings.paddle_speed),
        _ => String::new(),
    }
}

pub struct SettingsController {
    selected_row: usize,
    label_ids: Vec<ObjectId>,
    value_ids: Vec<ObjectId>,
    settings: Rc<RefCell<Settings>>,
    tinted_material: MaterialHandle,
    /// Kept alive so menu→settings→menu round trips cache-hit on the atlas.
    atlas: FontAtlas,
}

impl SettingsController {
    fn apply_selection(&self, scene: &mut Scene, gm: &mut GraphicsManager) {
        for (i, (&label_id, &value_id)) in
            self.label_ids.iter().zip(self.value_ids.iter()).enumerate()
        {
            let selected = i == self.selected_row;
            let (color, scale) = if selected {
                (COLOR_WHITE, ROW_SCALE_SELECTED)
            } else {
                (COLOR_GREY, ROW_SCALE_NORMAL)
            };
            for id in [label_id, value_id] {
                if let Some(obj) = scene.get_mut(id) {
                    obj.transform.scale = Vec3::splat(scale);
                }
                if let Some(b) = scene.behaviour_mut::<TintedTextLabelBehaviour>(id) {
                    b.set_color(gm, color);
                }
            }
        }
    }

    /// Apply an adjustment to the selected row and rebuild that row's value
    /// label text.
    fn adjust(&self, row: usize, delta: i32, scene: &mut Scene, gm: &mut GraphicsManager) {
        {
            let mut s = self.settings.borrow_mut();
            match (row, delta.signum()) {
                (0, 1) => s.inc_winning_score(),
                (0, -1) => s.dec_winning_score(),
                (1, 1) => s.inc_ball_speed(),
                (1, -1) => s.dec_ball_speed(),
                (2, 1) => s.inc_paddle_speed(),
                (2, -1) => s.dec_paddle_speed(),
                _ => {}
            }
        }
        let new_text = format_value(row, &self.settings.borrow());
        if let Some(b) =
            scene.behaviour_mut::<TintedTextLabelBehaviour>(self.value_ids[row])
        {
            b.set_text(gm, &self.atlas, self.tinted_material, &new_text, FONT_WORLD_SCALE);
        }
    }
}

impl Behaviour for SettingsController {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        if ctx.input.was_just_pressed(KeyCode::Escape) {
            self.settings.borrow().save();
            let s = Rc::clone(&self.settings);
            ctx.request_scene(move |res, gm| crate::scene_menu::build_menu(res, gm, s));
            return;
        }

        let n = self.label_ids.len();
        if ctx.input.was_just_pressed(KeyCode::ArrowUp) && n > 0 {
            self.selected_row = (self.selected_row + n - 1) % n;
        }
        if ctx.input.was_just_pressed(KeyCode::ArrowDown) && n > 0 {
            self.selected_row = (self.selected_row + 1) % n;
        }
        if ctx.input.was_just_pressed(KeyCode::ArrowLeft) {
            self.adjust(self.selected_row, -1, ctx.scene, ctx.graphics);
        }
        if ctx.input.was_just_pressed(KeyCode::ArrowRight) {
            self.adjust(self.selected_row, 1, ctx.scene, ctx.graphics);
        }

        self.apply_selection(ctx.scene, ctx.graphics);
    }
}
