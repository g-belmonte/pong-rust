//! Minimal HUD: a centred top-of-screen text label rendered against a
//! camera-slot-1 textured material.
//!
//! Exercises the multi-camera path end-to-end: the cube draws against
//! `scene.cameras[0]` (Camera3D), the label draws against `scene.cameras[1]`
//! (Camera2D), and `DepthMode::Disabled` on the HUD material guarantees the
//! glyphs paint on top of the 3D geometry regardless of depth values.
//!
//! Built as a slim cousin of `crates/pong/src/text.rs` (shelf-packed font
//! atlas → RGBA texture → per-glyph instances of a textured material). The
//! engine doesn't ship a text subsystem yet (that's Phase 9 territory); this
//! file inlines just enough to demonstrate the HUD path.

use std::any::Any;
use std::collections::HashMap;

use engine::graphics_manager::structures::hidden_transform;
use engine::graphics_manager::{
    Binding, DepthMode, GraphicsManager, MaterialDesc, MaterialHandle, ModelHandle, TextureHandle,
    VertexAttr,
};
use engine::resources::{Resources, Texture};
use engine::scene::Behaviour;
use engine::{Mat4, Vec3};
use fontdue::{Font, FontSettings};

const ATLAS_SIZE: u32 = 512;
const ATLAS_PADDING: u32 = 1;

#[derive(Clone, Copy)]
struct GlyphInfo {
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    width: f32,
    height: f32,
    xmin: f32,
    ymin: f32,
    advance: f32,
}

pub struct FontAtlas {
    pub texture: Texture,
    glyphs: HashMap<char, GlyphInfo>,
    ascent: f32,
    descent: f32,
}

impl FontAtlas {
    /// Rasterises printable ASCII at `px`, shelf-packs into an ATLAS_SIZE
    /// RGBA8 buffer (white RGB + alpha = bitmap), uploads via the engine's
    /// RAII texture loader.
    pub fn build(resources: &mut Resources, gm: &mut GraphicsManager, font_bytes: &[u8], px: f32) -> Self {
        let font = Font::from_bytes(font_bytes, FontSettings::default())
            .expect("Failed to load HUD font");

        let mut rastered: Vec<(char, fontdue::Metrics, Vec<u8>)> = Vec::new();
        for cp in 32u32..=126 {
            if let Some(ch) = std::char::from_u32(cp) {
                let (metrics, bitmap) = font.rasterize(ch, px);
                rastered.push((ch, metrics, bitmap));
            }
        }

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
                "HUD font atlas too small at {px}px"
            );
            origin.insert(*ch, (shelf_x, shelf_y));
            shelf_x += w + ATLAS_PADDING;
            shelf_h = shelf_h.max(h);
        }

        for (ch, metrics, bitmap) in &rastered {
            if metrics.width == 0 || metrics.height == 0 {
                continue;
            }
            let (ax, ay) = origin[ch];
            for row in 0..metrics.height {
                for col in 0..metrics.width {
                    let alpha = bitmap[row * metrics.width + col];
                    let dst = ((ay as usize + row) * ATLAS_SIZE as usize + (ax as usize + col)) * 4;
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
            .expect("font has no horizontal line metrics");

        FontAtlas { texture, glyphs, ascent: lm.ascent, descent: lm.descent }
    }
}

/// Register a HUD-textured material: same shaders as the engine's built-in
/// textured material, but sampling **camera slot 1** and running
/// depth-disabled so the text always paints on top of the cube.
///
/// The built-in textured material is hardcoded to slot 0 and can't be
/// retargeted; registering a parallel material is the supported path.
pub fn register_hud_material(resources: &mut Resources, gm: &mut GraphicsManager) -> MaterialHandle {
    resources.load_material(
        gm,
        &MaterialDesc {
            vertex_spv: include_bytes!("../../engine/shaders/spv/textured.vert.spv"),
            fragment_spv: include_bytes!("../../engine/shaders/spv/textured.frag.spv"),
            vertex_attrs: &[VertexAttr::F32x2],
            instance_attrs: &[VertexAttr::Mat4, VertexAttr::F32x2, VertexAttr::F32x2],
            bindings: &[Binding::CameraUbo(1), Binding::Sampler2d],
            depth: DepthMode::Disabled,
        },
    )
}

struct GlyphInstance {
    handle: ModelHandle,
    local_offset: Vec3,
    world_size: (f32, f32),
}

/// Multi-instance label, painted with `hud_material` against `atlas_texture`.
/// `scale` converts atlas pixels to HUD world units (Camera2D in slot 1).
pub struct HudLabelBehaviour {
    // Hold the RAII Texture so the atlas outlives every glyph instance.
    _atlas: FontAtlas,
    glyphs: Vec<GlyphInstance>,
}

impl HudLabelBehaviour {
    pub fn new(
        gm: &mut GraphicsManager,
        atlas: FontAtlas,
        hud_material: MaterialHandle,
        text: &str,
        scale: f32,
    ) -> Self {
        let total_advance: f32 = text
            .chars()
            .filter_map(|ch| atlas.glyphs.get(&ch))
            .map(|g| g.advance)
            .sum();

        // Centre the line on the parent transform: place baseline so
        // (ascent..descent) straddles label-local y=0 in atlas-pixel space
        // (positive Y is downwards, matching Pong's text.rs convention).
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
                let handle = register_glyph_instance(
                    gm,
                    hud_material,
                    atlas.texture.handle(),
                    uv_offset,
                    uv_scale,
                );
                glyphs.push(GlyphInstance {
                    handle,
                    local_offset: Vec3::new(tlx + w * 0.5, tly + h * 0.5, 0.0),
                    world_size: (w, h),
                });
            }
            pen_x += g.advance;
        }

        Self { _atlas: atlas, glyphs }
    }
}

fn register_glyph_instance(
    gm: &mut GraphicsManager,
    material: MaterialHandle,
    texture: TextureHandle,
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
) -> ModelHandle {
    // Pack uv_offset + uv_scale into the material's 16-byte "extra" payload,
    // matching textured.vert's `inUVOffset`@5 + `inUVScale`@6.
    let mut bytes = [0u8; 16];
    bytes[0..8].copy_from_slice(unsafe {
        ::std::slice::from_raw_parts(uv_offset.as_ptr() as *const u8, 8)
    });
    bytes[8..16].copy_from_slice(unsafe {
        ::std::slice::from_raw_parts(uv_scale.as_ptr() as *const u8, 8)
    });
    gm.register_material_instance(material, None, Some(texture), &bytes)
}

impl Behaviour for HudLabelBehaviour {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }

    fn collect_renderables(&self, parent_matrix: Mat4, out: &mut Vec<(ModelHandle, Mat4)>) {
        for g in &self.glyphs {
            let m = parent_matrix
                * Mat4::from_translation(g.local_offset)
                * Mat4::from_scale(Vec3::new(g.world_size.0, g.world_size.1, 1.0));
            out.push((g.handle, m));
        }
        // No visibility toggle — the HUD label in this demo is always on.
        // The hidden-transform helper is imported for parity with Pong's
        // text.rs; uncomment if the demo grows a toggle.
        let _ = hidden_transform;
    }

    fn on_despawn(&mut self, gm: &mut GraphicsManager) {
        for g in &self.glyphs {
            gm.unregister_textured_instance(g.handle);
        }
    }
}
