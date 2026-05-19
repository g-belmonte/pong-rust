//! In-match pause overlay.
//!
//! Spawned by [`crate::scene_game::build_game`] and hidden by default. ESC
//! during the game scene toggles [`engine::scene::Scene::set_paused`] and
//! flips the overlay's `visible` flag; while paused the engine suppresses
//! `fixed_update` (so the ball/paddles freeze), but keeps running `update`
//! so this controller can read input.
//!
//! Two options: **Continue** resumes play; **Quit** returns to the main
//! menu (the application Quit lives on the main menu's own Quit option).
//! Navigation matches the main menu: Up/Down + Enter, or mouse hover + left
//! click. While unpaused this behaviour does nothing except watch for the
//! ESC edge.
//!
//! ## Render layering
//!
//! The semi-transparent backdrop uses pong's custom `dimmer` material
//! (alpha-blended, see `crates/pong/shaders/src/dimmer.{vert,frag}`).
//! [`crate::scene_game::build_game`] registers the dimmer material *before*
//! the tinted-text material, so:
//!
//! ```text
//! solid (paddles/walls/digits)
//!   → textured (ball, welcome / game-over labels)
//!     → dimmer (this overlay's backdrop)
//!       → text_tint (this overlay's title + options)
//! ```
//!
//! Both built-ins and both pong materials run `DepthMode::Disabled`, so the
//! sequence is purely registration-order. Move the dimmer registration after
//! `text_tint` and the backdrop will paint over the options.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use engine::graphics_manager::{
    Binding, BlendMode, DepthMode, GraphicsManager, MaterialDesc, MaterialHandle, VertexAttr,
};
use engine::input::{KeyCode, MouseButton};
use engine::resources::{FontAtlas, Mesh, Resources};
use engine::scene::{Behaviour, Object, ObjectId, Renderable, Scene, Transform, UpdateCtx};
use engine::{Vec2, Vec3};

use crate::settings::Settings;
use crate::tinted_text::TintedTextLabelBehaviour;

/// Per-instance extra payload for the dimmer material:
/// `vec4 inColor` (16 bytes), straight RGBA.
const DIMMER_EXTRA_BYTES: usize = 16;

const COLOR_WHITE: [f32; 3] = [1.0, 1.0, 1.0];
const COLOR_GREY: [f32; 3] = [0.45, 0.45, 0.45];
/// Backdrop tint. 70 % alpha black — dark enough to read the menu against
/// the gameplay underneath, transparent enough that you still see the
/// paused frame.
const DIMMER_RGBA: [f32; 4] = [0.0, 0.0, 0.0, 0.7];
/// Dimmer quad side length in world units. Camera half-height is ~4.14 so a
/// 100×100 quad covers any sane aspect ratio.
const DIMMER_SIDE: f32 = 100.0;

const TITLE_SCALE: f32 = 1.8;
const OPTION_SCALE_NORMAL: f32 = 1.0;
const OPTION_SCALE_SELECTED: f32 = 1.25;

const TITLE_Y: f32 = -1.5;
const CONTINUE_Y: f32 = 0.0;
const QUIT_Y: f32 = 1.0;

const TITLE_TEXT: &str = "PAUSED";
const OPTION_TEXTS: [&str; 2] = ["Continue", "Quit"];

/// Register the dimmer material — a semi-transparent solid-colour quad.
///
/// Must be registered **before** the tinted-text material in any scene that
/// uses both (registration order = paint order for depth-disabled materials).
/// `crates/pong/src/scene_game.rs` is the only caller today.
pub fn register_dimmer_material(
    resources: &mut Resources,
    gm: &mut GraphicsManager,
) -> MaterialHandle {
    resources.load_material(
        gm,
        &MaterialDesc {
            vertex_spv: include_bytes!("../shaders/spv/dimmer.vert.spv"),
            fragment_spv: include_bytes!("../shaders/spv/dimmer.frag.spv"),
            vertex_attrs: &[VertexAttr::F32x2],
            instance_attrs: &[VertexAttr::Mat4, VertexAttr::F32x4],
            bindings: &[Binding::CameraUbo(0)],
            depth: DepthMode::Disabled,
            blend: BlendMode::Alpha,
        },
    )
}

fn pack_dimmer_extra(rgba: [f32; 4]) -> [u8; DIMMER_EXTRA_BYTES] {
    let mut bytes = [0u8; DIMMER_EXTRA_BYTES];
    bytes.copy_from_slice(unsafe { &::std::mem::transmute::<[f32; 4], [u8; 16]>(rgba) });
    bytes
}

/// Spawn the pause overlay (backdrop + "PAUSED" title + "Continue" / "Quit"
/// options) hidden, and the [`PauseController`] that drives it. Returns
/// nothing — the controller lives on its own Object and finds the overlay
/// pieces through their captured `ObjectId`s.
///
/// Layering note: the supplied `dimmer_mesh` is a 1×1 quad scaled up via the
/// Object's transform to cover the visible area at any aspect ratio. Both
/// the dimmer material and the tinted-text material run depth-disabled, so
/// the registration order in `register_*` above decides which paints on top.
#[allow(clippy::too_many_arguments)]
pub fn spawn(
    scene: &mut Scene,
    gm: &mut GraphicsManager,
    atlas: FontAtlas,
    dimmer_material: MaterialHandle,
    dimmer_mesh: Mesh,
    tinted_material: MaterialHandle,
    font_world_scale: f32,
    camera_half_height: f32,
    settings: Rc<RefCell<Settings>>,
) {
    // Backdrop: a large quad parented by the controller. Hidden initially —
    // visibility flips via `Object::visible` when pause toggles.
    let dimmer = scene.spawn(
        Object::new()
            .with_transform(Transform {
                position: Vec3::ZERO,
                scale: Vec3::new(DIMMER_SIDE, DIMMER_SIDE, 1.0),
                ..Transform::default()
            })
            .with_visible(false)
            .with_renderable(Renderable::Material {
                material: dimmer_material,
                mesh: Some(dimmer_mesh.handle()),
                texture: None,
                instance_data: pack_dimmer_extra(DIMMER_RGBA).to_vec(),
            }),
    );

    let title = scene.spawn(
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
                TITLE_TEXT,
                font_world_scale,
                COLOR_WHITE,
                false,
            )),
    );

    let continue_label = scene.spawn(
        Object::new()
            .with_transform(Transform {
                position: Vec3::new(0.0, CONTINUE_Y, 0.0),
                scale: Vec3::splat(OPTION_SCALE_SELECTED),
                ..Transform::default()
            })
            .with_behaviour(TintedTextLabelBehaviour::new(
                gm,
                &atlas,
                tinted_material,
                OPTION_TEXTS[0],
                font_world_scale,
                COLOR_WHITE,
                false,
            )),
    );

    let quit_label = scene.spawn(
        Object::new()
            .with_transform(Transform {
                position: Vec3::new(0.0, QUIT_Y, 0.0),
                scale: Vec3::splat(OPTION_SCALE_NORMAL),
                ..Transform::default()
            })
            .with_behaviour(TintedTextLabelBehaviour::new(
                gm,
                &atlas,
                tinted_material,
                OPTION_TEXTS[1],
                font_world_scale,
                COLOR_GREY,
                false,
            )),
    );

    let option_rects = PauseController::compute_option_rects(
        &atlas,
        font_world_scale,
        [(CONTINUE_Y, OPTION_TEXTS[0]), (QUIT_Y, OPTION_TEXTS[1])],
    );

    scene.spawn(Object::new().with_behaviour(PauseController::new(
        dimmer,
        title,
        continue_label,
        quit_label,
        option_rects,
        Vec2::ZERO,
        camera_half_height,
        settings,
        dimmer_mesh,
        atlas,
    )));
}

/// Pause-overlay controller. Drives ESC pause/resume edge detection, menu
/// selection (keyboard + mouse), and the Quit-back-to-menu transition.
pub struct PauseController {
    /// Backdrop object — `visible` flipped to follow the pause flag.
    dimmer: ObjectId,
    /// Title (PAUSED) and option labels.
    title: ObjectId,
    options: [ObjectId; 2],
    /// World-space hit-test rect for each option `[x_min, x_max, y_min, y_max]`.
    option_rects: [[f32; 4]; 2],

    selected: usize,
    last_mouse: Option<(f32, f32)>,
    camera_centre: Vec2,
    camera_half_height: f32,

    /// Handed back to `scene_menu::build_menu` on Quit.
    settings: Rc<RefCell<Settings>>,

    /// RAII handles kept alive for the scene lifetime so the registered
    /// renderable instances don't outlive their backing GPU resources.
    /// Underscore-prefixed: referenced only by handle through the registered
    /// dimmer instance, never read directly here.
    _dimmer_mesh: Mesh,
    _atlas: FontAtlas,
}

impl PauseController {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dimmer: ObjectId,
        title: ObjectId,
        continue_label: ObjectId,
        quit_label: ObjectId,
        option_rects: [[f32; 4]; 2],
        camera_centre: Vec2,
        camera_half_height: f32,
        settings: Rc<RefCell<Settings>>,
        dimmer_mesh: Mesh,
        atlas: FontAtlas,
    ) -> Self {
        Self {
            dimmer,
            title,
            options: [continue_label, quit_label],
            option_rects,
            selected: 0,
            last_mouse: None,
            camera_centre,
            camera_half_height,
            settings,
            _dimmer_mesh: dimmer_mesh,
            _atlas: atlas,
        }
    }

    /// Compute the world-space hit-test rect for the "Continue" / "Quit"
    /// options, sized against the supplied font atlas. Caller picks the row
    /// Y; widths are derived from the glyph advances so the rects track the
    /// actual rendered label.
    pub fn compute_option_rects(
        atlas: &FontAtlas,
        scale: f32,
        rows: [(f32, &str); 2],
    ) -> [[f32; 4]; 2] {
        let mut rects = [[0.0; 4]; 2];
        for (i, (y, text)) in rows.iter().enumerate() {
            let (w, h) = label_world_size(atlas, text, scale);
            // Hit-test at the *selected* scale so the cursor doesn't fall off
            // the edge of an option as it grows on hover.
            let half_w = (w / 2.0) * OPTION_SCALE_SELECTED;
            let half_h = (h / 2.0) * OPTION_SCALE_SELECTED;
            rects[i] = [-half_w, half_w, y - half_h, y + half_h];
        }
        rects
    }

    fn apply_selection(&self, scene: &mut Scene, gm: &mut GraphicsManager) {
        for (i, &id) in self.options.iter().enumerate() {
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

    fn set_overlay_visible(&self, scene: &mut Scene, visible: bool) {
        if let Some(obj) = scene.get_mut(self.dimmer) {
            obj.visible = visible;
        }
        for id in [self.title, self.options[0], self.options[1]] {
            if let Some(label) = scene.behaviour_mut::<TintedTextLabelBehaviour>(id) {
                label.set_visible(visible);
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

    /// Reset selection + cached mouse state so a reopen of the overlay starts
    /// fresh (Continue highlighted, cursor position re-acquired on next move).
    fn reset_selection_state(&mut self) {
        self.selected = 0;
        self.last_mouse = None;
    }
}

impl Behaviour for PauseController {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn update(&mut self, ctx: &mut UpdateCtx) {
        let was_paused = ctx.scene.is_paused();

        // ESC: toggle pause. Unpause path also flips visibility back off.
        if ctx.input.was_just_pressed(KeyCode::Escape) {
            if was_paused {
                ctx.scene.set_paused(false);
                self.set_overlay_visible(ctx.scene, false);
            } else {
                ctx.scene.set_paused(true);
                self.reset_selection_state();
                self.set_overlay_visible(ctx.scene, true);
                self.apply_selection(ctx.scene, ctx.graphics);
            }
            return;
        }

        if !was_paused {
            return;
        }

        // --- Paused-only input handling ---

        let n = self.options.len();
        if ctx.input.was_just_pressed(KeyCode::ArrowUp) && n > 0 {
            self.selected = (self.selected + n - 1) % n;
        }
        if ctx.input.was_just_pressed(KeyCode::ArrowDown) && n > 0 {
            self.selected = (self.selected + 1) % n;
        }

        // Mouse hover: only honour actual movement, same rationale as the main
        // menu (a stationary cursor over a non-selected option shouldn't
        // override keyboard nav after the player reopened the overlay).
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
                0 => {
                    // Continue: unpause and hide the overlay.
                    ctx.scene.set_paused(false);
                    self.set_overlay_visible(ctx.scene, false);
                }
                1 => {
                    // Quit: back to main menu. Don't bother unpausing — the
                    // scene swap drops this Scene entirely.
                    let s = Rc::clone(&self.settings);
                    ctx.request_scene(move |res, gm| {
                        crate::scene_menu::build_menu(res, gm, s)
                    });
                }
                _ => {}
            }
        }
    }
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

