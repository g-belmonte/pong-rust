use std::any::Any;

use engine::{Mat4, Vec3};

use engine::graphics_manager::structures::hidden_transform;
use engine::graphics_manager::{GraphicsManager, ModelHandle};
use engine::resources::FontAtlas;
use engine::scene::Behaviour;

// Per-glyph state. `local_offset` is the centre of the glyph in label-local
// world coords; `world_size` is the glyph's scaled width/height. The instance
// handle drives one of the textured-pipeline batch entries — every glyph in
// every label sharing the same atlas folds into a single `vkCmdDrawIndexed`.
struct GlyphInstance {
    handle: ModelHandle,
    local_offset: Vec3,
    world_size: (f32, f32),
}

/// Multi-instance label behaviour. The parent `Object`'s transform positions
/// the label as a whole; per-glyph local offsets + sizes compose against it
/// in `collect_renderables`. Visibility uses the engine-standard parking trick.
pub struct TextLabelBehaviour {
    glyphs: Vec<GlyphInstance>,
    pub visible: bool,
}

impl TextLabelBehaviour {
    /// `scale` converts atlas pixels to world units (e.g. 0.005 ≈ 200 px/world).
    pub fn new(
        gm: &mut GraphicsManager,
        atlas: &FontAtlas,
        text: &str,
        scale: f32,
        visible: bool,
    ) -> Self {
        let total_advance: f32 = text
            .chars()
            .filter_map(|ch| atlas.glyphs.get(&ch))
            .map(|g| g.advance)
            .sum();

        // Vertically centre the line on the parent transform: place baseline
        // so (ascent..descent) straddles y=0 in label-local pixel space
        // (positive Y is downwards).
        let baseline_y_local = (atlas.ascent + atlas.descent) / 2.0;
        let mut pen_x = -total_advance / 2.0;

        let mut glyphs = Vec::new();
        for ch in text.chars() {
            let g = match atlas.glyphs.get(&ch) {
                Some(g) => *g,
                None => continue,
            };
            if g.width > 0.0 && g.height > 0.0 {
                // fontdue: ymin is the bitmap's bottom relative to baseline, up-positive.
                // Top of bitmap above baseline = ymin + height. In down-positive label
                // coords: top_y = baseline_y_local - (ymin + height).
                let tlx = (pen_x + g.xmin) * scale;
                let tly = (baseline_y_local - (g.ymin + g.height)) * scale;
                let w = g.width * scale;
                let h = g.height * scale;

                let uv_offset = g.uv_min;
                let uv_scale = [g.uv_max[0] - g.uv_min[0], g.uv_max[1] - g.uv_min[1]];
                let handle =
                    gm.register_textured_instance(atlas.texture.handle(), uv_offset, uv_scale);

                let local_offset = Vec3::new(tlx + w * 0.5, tly + h * 0.5, 0.0);
                glyphs.push(GlyphInstance {
                    handle,
                    local_offset,
                    world_size: (w, h),
                });
            }
            pen_x += g.advance;
        }

        Self { glyphs, visible }
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }
}

impl Behaviour for TextLabelBehaviour {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn collect_renderables(
        &self,
        parent_matrix: Mat4,
        out: &mut Vec<(ModelHandle, Mat4)>,
    ) {
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
            gm.unregister_textured_instance(g.handle);
        }
    }
}
