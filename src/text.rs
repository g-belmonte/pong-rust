use std::collections::HashMap;

use cgmath::{Matrix4, Vector3};
use fontdue::{Font, FontSettings};

use crate::graphics_manager::structures::hidden_transform;
use crate::graphics_manager::{GraphicsManager, ModelHandle, TextureHandle};

const PRINTABLE_ASCII_START: u32 = 32;
const PRINTABLE_ASCII_END: u32 = 126;
const ATLAS_SIZE: u32 = 512;
const ATLAS_PADDING: u32 = 1;

#[derive(Clone, Copy)]
pub struct GlyphInfo {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub width: f32,
    pub height: f32,
    pub xmin: f32,
    // Bottom of bitmap relative to baseline, fontdue's up-positive convention.
    pub ymin: f32,
    pub advance: f32,
}

pub struct FontAtlas {
    pub texture: TextureHandle,
    pub glyphs: HashMap<char, GlyphInfo>,
    pub ascent: f32,
    pub descent: f32,
}

impl FontAtlas {
    pub fn build(gm: &mut GraphicsManager, font_bytes: &[u8], px: f32) -> Self {
        let font = Font::from_bytes(font_bytes, FontSettings::default())
            .expect("Failed to load font");

        let mut rastered: Vec<(char, fontdue::Metrics, Vec<u8>)> = Vec::new();
        for cp in PRINTABLE_ASCII_START..=PRINTABLE_ASCII_END {
            if let Some(ch) = std::char::from_u32(cp) {
                let (metrics, bitmap) = font.rasterize(ch, px);
                rastered.push((ch, metrics, bitmap));
            }
        }

        // Shelf packing: tallest glyphs first, fill rows greedily.
        let mut order: Vec<usize> = (0..rastered.len()).collect();
        order.sort_by(|&a, &b| rastered[b].1.height.cmp(&rastered[a].1.height));

        let mut atlas = vec![0u8; (ATLAS_SIZE * ATLAS_SIZE * 4) as usize];
        let mut origin: HashMap<char, (u32, u32)> = HashMap::new();

        let mut shelf_x: u32 = 0;
        let mut shelf_y: u32 = 0;
        let mut shelf_h: u32 = 0;

        for &i in &order {
            let (ch, metrics, _) = &rastered[i];
            let w = metrics.width as u32;
            let h = metrics.height as u32;
            if w == 0 || h == 0 {
                origin.insert(*ch, (0, 0));
                continue;
            }
            if shelf_x + w + ATLAS_PADDING > ATLAS_SIZE {
                shelf_y += shelf_h + ATLAS_PADDING;
                shelf_x = 0;
                shelf_h = 0;
            }
            assert!(
                shelf_y + h <= ATLAS_SIZE,
                "font atlas {}x{} too small for printable ASCII at {}px",
                ATLAS_SIZE, ATLAS_SIZE, px
            );
            origin.insert(*ch, (shelf_x, shelf_y));
            shelf_x += w + ATLAS_PADDING;
            shelf_h = shelf_h.max(h);
        }

        // White RGB + alpha = bitmap. Lets the existing R8G8B8A8_SRGB textured
        // pipeline render glyphs without any shader changes.
        for (ch, metrics, bitmap) in &rastered {
            if metrics.width == 0 || metrics.height == 0 {
                continue;
            }
            let (ax, ay) = origin[ch];
            for row in 0..metrics.height {
                for col in 0..metrics.width {
                    let alpha = bitmap[row * metrics.width + col];
                    let dx = ax as usize + col;
                    let dy = ay as usize + row;
                    let dst = (dy * ATLAS_SIZE as usize + dx) * 4;
                    atlas[dst] = 255;
                    atlas[dst + 1] = 255;
                    atlas[dst + 2] = 255;
                    atlas[dst + 3] = alpha;
                }
            }
        }

        let texture = gm.register_texture_rgba(ATLAS_SIZE, ATLAS_SIZE, &atlas);

        let mut glyphs = HashMap::new();
        let s = ATLAS_SIZE as f32;
        for (ch, metrics, _) in &rastered {
            let (ax, ay) = origin[ch];
            let w = metrics.width as f32;
            let h = metrics.height as f32;
            glyphs.insert(*ch, GlyphInfo {
                uv_min: [ax as f32 / s, ay as f32 / s],
                uv_max: [(ax as f32 + w) / s, (ay as f32 + h) / s],
                width: w,
                height: h,
                xmin: metrics.xmin as f32,
                ymin: metrics.ymin as f32,
                advance: metrics.advance_width,
            });
        }

        let line_metrics = font
            .horizontal_line_metrics(px)
            .expect("font has no horizontal line metrics");

        FontAtlas {
            texture,
            glyphs,
            ascent: line_metrics.ascent,
            descent: line_metrics.descent,
        }
    }
}

// Per-glyph state. The instance handle drives a single textured-instance draw
// (every glyph in the label shares the atlas texture, so they batch into one
// `vkCmdDrawIndexed`). `local_offset` is the centre of the glyph in label-local
// world coords; `world_size` is the glyph's scaled width/height.
struct GlyphInstance {
    handle: ModelHandle,
    local_offset: Vector3<f32>,
    world_size: (f32, f32),
}

pub struct TextLabel {
    glyphs: Vec<GlyphInstance>,
    pub position: Vector3<f32>,
    pub visible: bool,
}

impl TextLabel {
    /// Build a label centred horizontally and vertically at `position`.
    /// `scale` converts atlas pixels to world units (e.g. 0.01 = 100 px / world unit).
    pub fn new(
        gm: &mut GraphicsManager,
        atlas: &FontAtlas,
        text: &str,
        position: Vector3<f32>,
        scale: f32,
    ) -> Self {
        let total_advance: f32 = text
            .chars()
            .filter_map(|ch| atlas.glyphs.get(&ch))
            .map(|g| g.advance)
            .sum();

        // Line vertically centred: baseline placed so the (ascent..descent) span
        // straddles y=0 in label-local pixel space (positive Y is downwards).
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
                let handle = gm.register_textured_instance(atlas.texture, uv_offset, uv_scale);

                // Centre of the glyph rect in label-local coords. The unit quad
                // is [-0.5..0.5]^2, so per-frame model = Translate(label_pos +
                // glyph_centre) * Scale(w, h) maps it onto the glyph's world rect.
                let local_offset = Vector3 {
                    x: tlx + w * 0.5,
                    y: tly + h * 0.5,
                    z: 0.0,
                };

                glyphs.push(GlyphInstance {
                    handle,
                    local_offset,
                    world_size: (w, h),
                });
            }
            pen_x += g.advance;
        }

        Self {
            glyphs,
            position,
            visible: true,
        }
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn get_model_transforms(&self) -> Vec<(ModelHandle, Matrix4<f32>)> {
        // Hidden labels park every glyph off-screen rather than dropping
        // the transforms — see `hidden_transform` for the rationale.
        self.glyphs
            .iter()
            .map(|g| {
                let m = if self.visible {
                    Matrix4::from_translation(self.position + g.local_offset)
                        * Matrix4::from_nonuniform_scale(g.world_size.0, g.world_size.1, 1.0)
                } else {
                    hidden_transform()
                };
                (g.handle, m)
            })
            .collect()
    }
}
