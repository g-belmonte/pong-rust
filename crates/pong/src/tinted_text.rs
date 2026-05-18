//! Multi-instance text label rendered through pong's custom tinted-text
//! material — same shape as [`crate::text::TextLabelBehaviour`] but each
//! glyph carries a per-instance RGB colour the fragment shader multiplies
//! into the sampled atlas.
//!
//! Used by the main menu to draw the highlighted option in white and the
//! others in dim grey. Selection changes call [`Self::set_color`], which
//! pokes the new colour into each registered instance's "extra" payload via
//! [`engine::graphics_manager::GraphicsManager::set_instance_extra`] — far
//! cheaper than unregister-and-rebuild per selection edge.

use std::any::Any;

use engine::{Mat4, Vec3};

use engine::graphics_manager::structures::hidden_transform;
use engine::graphics_manager::{GraphicsManager, MaterialHandle, ModelHandle};
use engine::resources::FontAtlas;
use engine::scene::Behaviour;

/// Per-glyph extra payload, matching `text_tint.vert`'s instance attributes:
/// `vec2 inUVOffset` (8B) + `vec2 inUVScale` (8B) + `vec3 inColor` (12B).
const EXTRA_BYTES: usize = 28;

struct GlyphInstance {
    handle: ModelHandle,
    local_offset: Vec3,
    world_size: (f32, f32),
    // Cached so `set_color` can rebuild the full extra without re-deriving
    // the glyph's atlas region.
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
}

/// Tinted-text variant of `TextLabelBehaviour`. Owns the per-instance colour
/// and lets game code mutate it through [`Self::set_color`].
pub struct TintedTextLabelBehaviour {
    glyphs: Vec<GlyphInstance>,
    color: [f32; 3],
    pub visible: bool,
}

impl TintedTextLabelBehaviour {
    /// Register one tinted glyph instance per visible character. `material`
    /// must have been registered with `vertex_attrs: [F32x2]`,
    /// `instance_attrs: [Mat4, F32x2, F32x2, F32x3]`,
    /// `bindings: [CameraUbo(0), Sampler2d]` (see
    /// [`crate::scene_menu::register_tinted_text_material`]).
    pub fn new(
        gm: &mut GraphicsManager,
        atlas: &FontAtlas,
        material: MaterialHandle,
        text: &str,
        scale: f32,
        color: [f32; 3],
        visible: bool,
    ) -> Self {
        let total_advance: f32 = text
            .chars()
            .filter_map(|ch| atlas.glyphs.get(&ch))
            .map(|g| g.advance)
            .sum();

        let baseline_y_local = (atlas.ascent + atlas.descent) / 2.0;
        let mut pen_x = -total_advance / 2.0;

        let mut glyphs = Vec::new();
        for ch in text.chars() {
            let g = match atlas.glyphs.get(&ch) {
                Some(g) => *g,
                None => continue,
            };
            if g.width > 0.0 && g.height > 0.0 {
                let tlx = (pen_x + g.xmin) * scale;
                let tly = (baseline_y_local - (g.ymin + g.height)) * scale;
                let w = g.width * scale;
                let h = g.height * scale;
                let uv_offset = g.uv_min;
                let uv_scale = [g.uv_max[0] - g.uv_min[0], g.uv_max[1] - g.uv_min[1]];
                let extra = pack_extra(uv_offset, uv_scale, color);
                let handle = gm.register_material_instance(
                    material,
                    None,
                    Some(atlas.texture.handle()),
                    &extra,
                );
                glyphs.push(GlyphInstance {
                    handle,
                    local_offset: Vec3::new(tlx + w * 0.5, tly + h * 0.5, 0.0),
                    world_size: (w, h),
                    uv_offset,
                    uv_scale,
                });
            }
            pen_x += g.advance;
        }

        Self { glyphs, color, visible }
    }

    /// Update the tint applied to every glyph. No-op if `color` matches the
    /// current value.
    pub fn set_color(&mut self, gm: &mut GraphicsManager, color: [f32; 3]) {
        if self.color == color {
            return;
        }
        self.color = color;
        for g in &self.glyphs {
            let extra = pack_extra(g.uv_offset, g.uv_scale, color);
            gm.set_instance_extra(g.handle, &extra);
        }
    }

    #[allow(dead_code)] // Public surface used by future menus (e.g. Settings).
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
}

impl Behaviour for TintedTextLabelBehaviour {
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn as_any(&self) -> &dyn Any { self }

    fn collect_renderables(&self, parent_matrix: Mat4, out: &mut Vec<(ModelHandle, Mat4)>) {
        for g in &self.glyphs {
            let m = if self.visible {
                parent_matrix
                    * Mat4::from_translation(g.local_offset)
                    * Mat4::from_scale(Vec3::new(g.world_size.0, g.world_size.1, 1.0))
            } else {
                hidden_transform()
            };
            out.push((g.handle, m));
        }
    }

    fn on_despawn(&mut self, gm: &mut GraphicsManager) {
        for g in &self.glyphs {
            gm.unregister_instance(g.handle);
        }
    }
}

fn pack_extra(uv_offset: [f32; 2], uv_scale: [f32; 2], color: [f32; 3]) -> [u8; EXTRA_BYTES] {
    let mut bytes = [0u8; EXTRA_BYTES];
    bytes[0..8].copy_from_slice(unsafe { &::std::mem::transmute::<[f32; 2], [u8; 8]>(uv_offset) });
    bytes[8..16].copy_from_slice(unsafe { &::std::mem::transmute::<[f32; 2], [u8; 8]>(uv_scale) });
    bytes[16..28].copy_from_slice(unsafe { &::std::mem::transmute::<[f32; 3], [u8; 12]>(color) });
    bytes
}
