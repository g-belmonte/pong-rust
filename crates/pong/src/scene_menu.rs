//! Main menu scene.
//!
//! Title "PONG" + two options: Play and Quit. Navigation by Up/Down arrows
//! and Enter, or by mouse hover + left click. Escape exits.
//!
//! Selection is shown two ways: the highlighted option is white and larger
//! (`OPTION_SCALE_SELECTED`); unselected options are dim grey at normal
//! size. Colour comes from the custom `text_tint` material; size comes from
//! the Object's `transform.scale`.
//!
//! Entered at program start (see `main.rs`) and again every time the game
//! scene's `PhaseController` requests a swap (Escape in-game, or Escape on
//! the Game Over screen). Shared assets (font atlas) cache-hit through
//! `Resources` across the swap — the menu's atlas survives into the game
//! scene and back without any GPU re-upload.

use std::any::Any;

use engine::camera::Camera2D;
use engine::graphics_manager::{
    Binding, DepthMode, GraphicsManager, MaterialDesc, MaterialHandle, VertexAttr,
};
use engine::input::{KeyCode, MouseButton};
use engine::resources::{FontAtlas, Resources};
use engine::scene::{Behaviour, Object, ObjectId, Scene, Transform, UpdateCtx};
use engine::{Vec2, Vec3};

use crate::scene_game;
use crate::tinted_text::TintedTextLabelBehaviour;

const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");
const FONT_RASTER_PX: f32 = 48.0;
// Atlas-pixel → world-unit conversion, matches scene_game so glyphs are the
// same physical size on screen across the swap.
const FONT_WORLD_SCALE: f32 = 0.005;

const TITLE_SCALE: f32 = 2.0;
const OPTION_SCALE_NORMAL: f32 = 1.0;
const OPTION_SCALE_SELECTED: f32 = 1.25;
const COLOR_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const COLOR_GREY: [f32; 3] = [0.45, 0.45, 0.45];

/// Same as `scene_game::CAMERA_HALF_HEIGHT` — keeps the menu camera matching
/// the game camera so swapping scenes doesn't re-letterbox the window.
const CAMERA_HALF_HEIGHT: f32 = 4.142;

const TITLE_Y: f32 = -2.4;
const OPTION_TEXTS: [&str; 2] = ["Play", "Quit"];
const OPTION_Y: [f32; 2] = [-0.3, 1.0];

/// Register the tinted-text material. Shaders ship as SPIR-V under
/// `crates/pong/shaders/spv/`; recompile via `scripts/compile-shaders.sh
/// crates/pong/shaders/src crates/pong/shaders/spv`.
fn register_tinted_text_material(
    resources: &mut Resources,
    gm: &mut GraphicsManager,
) -> MaterialHandle {
    resources.load_material(
        gm,
        &MaterialDesc {
            vertex_spv: include_bytes!("../shaders/spv/text_tint.vert.spv"),
            fragment_spv: include_bytes!("../shaders/spv/text_tint.frag.spv"),
            vertex_attrs: &[VertexAttr::F32x2],
            instance_attrs: &[
                VertexAttr::Mat4,
                VertexAttr::F32x2,
                VertexAttr::F32x2,
                VertexAttr::F32x3,
            ],
            bindings: &[Binding::CameraUbo(0), Binding::Sampler2d],
            depth: DepthMode::Disabled,
        },
    )
}

pub fn build_menu(resources: &mut Resources, gm: &mut GraphicsManager) -> Scene {
    let mut scene = Scene::new();
    scene.set_camera(0, Box::new(Camera2D::new(Vec2::ZERO, CAMERA_HALF_HEIGHT)));

    let atlas = resources.load_font(gm, FONT_BYTES, FONT_RASTER_PX);
    let tinted_material = register_tinted_text_material(resources, gm);

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
                "PONG",
                FONT_WORLD_SCALE,
                COLOR_WHITE,
                true,
            )),
    );

    let mut option_ids: Vec<ObjectId> = Vec::new();
    let mut option_rects: Vec<[f32; 4]> = Vec::new();
    for (i, &text) in OPTION_TEXTS.iter().enumerate() {
        let (w, h) = label_world_size(&atlas, text, FONT_WORLD_SCALE);
        // Hit-test box is computed at the *selected* (larger) scale so the
        // cursor doesn't fall off the edge of an option as it grows on hover.
        let half_w = (w / 2.0) * OPTION_SCALE_SELECTED;
        let half_h = (h / 2.0) * OPTION_SCALE_SELECTED;
        option_rects.push([
            -half_w,
            half_w,
            OPTION_Y[i] - half_h,
            OPTION_Y[i] + half_h,
        ]);

        let (initial_color, initial_scale) = if i == 0 {
            (COLOR_WHITE, OPTION_SCALE_SELECTED)
        } else {
            (COLOR_GREY, OPTION_SCALE_NORMAL)
        };
        let id = scene.spawn(
            Object::new()
                .with_transform(Transform {
                    position: Vec3::new(0.0, OPTION_Y[i], 0.0),
                    scale: Vec3::splat(initial_scale),
                    ..Transform::default()
                })
                .with_behaviour(TintedTextLabelBehaviour::new(
                    gm,
                    &atlas,
                    tinted_material,
                    text,
                    FONT_WORLD_SCALE,
                    initial_color,
                    true,
                )),
        );
        option_ids.push(id);
    }

    scene.spawn(Object::new().with_behaviour(MenuController {
        selected: 0,
        option_ids,
        option_rects,
        last_mouse: None,
        camera_centre: Vec2::ZERO,
        camera_half_height: CAMERA_HALF_HEIGHT,
        _atlas: atlas,
    }));

    scene
}

fn label_world_size(atlas: &FontAtlas, text: &str, scale: f32) -> (f32, f32) {
    let total_advance: f32 = text
        .chars()
        .filter_map(|ch| atlas.glyphs.get(&ch))
        .map(|g| g.advance)
        .sum();
    let line_h = (atlas.ascent - atlas.descent) * scale;
    (total_advance * scale, line_h)
}

/// Owns the menu's selection state and routes keyboard + mouse input.
pub struct MenuController {
    selected: usize,
    option_ids: Vec<ObjectId>,
    /// Per-option world-space rect `[x_min, x_max, y_min, y_max]`.
    option_rects: Vec<[f32; 4]>,
    /// Last observed mouse position. Used to detect movement so a stationary
    /// cursor over a non-selected option doesn't override keyboard nav.
    last_mouse: Option<(f32, f32)>,
    camera_centre: Vec2,
    camera_half_height: f32,
    /// Kept alive so the atlas's [`engine::resources::Texture`] survives a
    /// menu→game→menu round trip — the game scene's `load_font` cache-hits
    /// against this Rc rather than re-uploading.
    _atlas: FontAtlas,
}

impl MenuController {
    fn apply_selection(&self, scene: &mut Scene, gm: &mut GraphicsManager) {
        for (i, &id) in self.option_ids.iter().enumerate() {
            let selected = i == self.selected;
            let (color, scale) = if selected {
                (COLOR_WHITE, OPTION_SCALE_SELECTED)
            } else {
                (COLOR_GREY, OPTION_SCALE_NORMAL)
            };
            if let Some(obj) = scene.get_mut(id) {
                obj.transform.scale = Vec3::splat(scale);
            }
            if let Some(label) = scene.behaviour_mut::<TintedTextLabelBehaviour>(id) {
                label.set_color(gm, color);
            }
        }
    }

    fn screen_to_world(&self, px: f32, py: f32, extent: [u32; 2]) -> Vec2 {
        let w = extent[0] as f32;
        let h = extent[1] as f32;
        let ndc_x = (px / w) * 2.0 - 1.0;
        let ndc_y = (py / h) * 2.0 - 1.0;
        let aspect = w / h;
        let half_w = self.camera_half_height * aspect;
        Vec2::new(
            self.camera_centre.x + ndc_x * half_w,
            self.camera_centre.y + ndc_y * self.camera_half_height,
        )
    }

    fn hit_test(&self, world: Vec2) -> Option<usize> {
        for (i, r) in self.option_rects.iter().enumerate() {
            if world.x >= r[0] && world.x <= r[1] && world.y >= r[2] && world.y <= r[3] {
                return Some(i);
            }
        }
        None
    }
}

impl Behaviour for MenuController {
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn as_any(&self) -> &dyn Any { self }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        // Escape from the menu = exit the program. Quit option does the same.
        if ctx.input.was_just_pressed(KeyCode::Escape) {
            ctx.request_exit();
            return;
        }

        let n = self.option_ids.len();
        if ctx.input.was_just_pressed(KeyCode::ArrowUp) && n > 0 {
            self.selected = (self.selected + n - 1) % n;
        }
        if ctx.input.was_just_pressed(KeyCode::ArrowDown) && n > 0 {
            self.selected = (self.selected + 1) % n;
        }

        // Mouse hover: only honour it on actual movement so keyboard nav
        // isn't overridden by a cursor sitting still over a non-selected
        // option (e.g. after the player tabbed back into the window).
        let mouse = ctx.input.mouse_position();
        let mouse_moved = self
            .last_mouse
            .map_or(true, |(lx, ly)| lx != mouse.0 || ly != mouse.1);
        self.last_mouse = Some(mouse);
        if mouse_moved {
            let world = self.screen_to_world(mouse.0, mouse.1, ctx.graphics.extent());
            if let Some(i) = self.hit_test(world) {
                self.selected = i;
            }
        }

        // Activation. Click activates only if the cursor is over the
        // currently-selected option (so a click in empty space does nothing).
        let enter = ctx.input.was_just_pressed(KeyCode::Enter);
        let clicked = ctx.input.was_mouse_just_pressed(MouseButton::Left);
        let click_hits_selected = if clicked {
            let world = self.screen_to_world(mouse.0, mouse.1, ctx.graphics.extent());
            self.hit_test(world) == Some(self.selected)
        } else {
            false
        };

        self.apply_selection(ctx.scene, ctx.graphics);

        if enter || click_hits_selected {
            match self.selected {
                0 => ctx.request_scene(scene_game::build_game),
                1 => ctx.request_exit(),
                _ => {}
            }
        }
    }
}
