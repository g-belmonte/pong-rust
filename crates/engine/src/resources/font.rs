//! Font atlas baking.
//!
//! [`FontAtlas::build`] rasterises printable ASCII at a given pixel size
//! using `fontdue`, shelf-packs the glyph bitmaps into one RGBA8 texture
//! (white RGB + alpha = bitmap mask — works against the engine's built-in
//! textured pipeline whose fragment shader `discard`s on `alpha < 0.5`), and
//! uploads it via the renderer's RAII texture loader.
//!
//! Game code talks to this through [`crate::resources::Resources::load_font`].
//! What the engine deliberately *doesn't* ship:
//!
//! - A text-label behaviour. Each game writes its own — Pong's `text.rs` and
//!   test-3d's `hud.rs` have different positioning conventions (world-space
//!   vs HUD-space) and material selections (built-in textured at slot 0 vs
//!   custom textured at slot 1). The atlas is the deduplicated bit; the
//!   per-glyph rendering policy stays game-side.
//! - Non-ASCII glyphs. Adding Unicode ranges is a constructor parameter
//!   away when a game needs it.
//! - Mip-mapped atlases / SDF / MSDF. The single-resolution shelf-packed
//!   atlas matches what Pong + test-3d need today.

use std::collections::HashMap;

use fontdue::{Font, FontSettings};

use crate::graphics_manager::GraphicsManager;
use crate::resources::{Resources, Texture};

/// Atlas raster dimensions (square, RGBA8). Tall fonts or large glyph sets
/// can overflow — the constructor asserts on overflow so the failure mode
/// is loud. Bump if the panic fires.
pub const ATLAS_SIZE: u32 = 512;
/// 1-pixel padding between glyphs so bilinear sampling can't bleed
/// neighbours into a glyph's edge.
const ATLAS_PADDING: u32 = 1;
const PRINTABLE_ASCII_START: u32 = 32;
const PRINTABLE_ASCII_END: u32 = 126;

/// Layout of one glyph inside the atlas, in atlas-pixel and font-pixel units.
///
/// `uv_min` / `uv_max` are in `[0, 1]` texture-space (atlas-relative).
/// `width` / `height` / `xmin` / `ymin` / `advance` are in *font-pixel* units
/// at the resolution the atlas was rasterised at — the same units `fontdue`
/// reports. `ymin` is fontdue's "bottom of bitmap above baseline" with the
/// `+Y up` convention.
#[derive(Clone, Copy, Debug)]
pub struct GlyphInfo {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub width: f32,
    pub height: f32,
    pub xmin: f32,
    pub ymin: f32,
    pub advance: f32,
}

/// Baked glyph atlas + the RGBA8 texture it lives in.
///
/// Owns the [`Texture`] via RAII — drop the atlas and the GPU resource
/// queues for cleanup like any other engine-loaded texture. `glyphs` is
/// keyed by char; lookups for missing chars return `None` (caller skips them).
pub struct FontAtlas {
    pub texture: Texture,
    pub glyphs: HashMap<char, GlyphInfo>,
    /// Font ascent at the rasterisation size (font pixels above the baseline,
    /// positive).
    pub ascent: f32,
    /// Font descent at the rasterisation size (font pixels below the
    /// baseline, *negative* — matches fontdue's convention).
    pub descent: f32,
}

impl FontAtlas {
    /// Rasterise printable ASCII at `px` and pack into a 512×512 atlas.
    /// Panics if the atlas overflows (font + size too large for one sheet)
    /// or if `font_bytes` doesn't parse — both are programmer errors that
    /// should fail loud at startup, not at runtime.
    pub fn build(
        resources: &mut Resources,
        gm: &mut GraphicsManager,
        font_bytes: &[u8],
        px: f32,
    ) -> Self {
        let font = Font::from_bytes(font_bytes, FontSettings::default())
            .expect("FontAtlas::build: failed to parse font bytes");

        let mut rastered: Vec<(char, fontdue::Metrics, Vec<u8>)> = Vec::new();
        for cp in PRINTABLE_ASCII_START..=PRINTABLE_ASCII_END {
            if let Some(ch) = std::char::from_u32(cp) {
                let (metrics, bitmap) = font.rasterize(ch, px);
                rastered.push((ch, metrics, bitmap));
            }
        }

        // Shelf-pack tallest glyphs first.
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
                "FontAtlas: {ATLAS_SIZE}x{ATLAS_SIZE} atlas too small for printable ASCII at {px}px"
            );
            origin.insert(*ch, (shelf_x, shelf_y));
            shelf_x += w + ATLAS_PADDING;
            shelf_h = shelf_h.max(h);
        }

        // White RGB + per-glyph alpha. Renders directly under the engine's
        // textured fragment shader (which discards `alpha < 0.5`).
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

        let texture = resources.load_texture_rgba(gm, ATLAS_SIZE, ATLAS_SIZE, &atlas);

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

        let lm = font
            .horizontal_line_metrics(px)
            .expect("FontAtlas::build: font has no horizontal line metrics");

        FontAtlas {
            texture,
            glyphs,
            ascent: lm.ascent,
            descent: lm.descent,
        }
    }
}
