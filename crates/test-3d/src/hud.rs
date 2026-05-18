//! Minimal HUD: a centred top-of-screen text label rendered against a
//! camera-slot-1 textured material.
//!
//! Exercises the multi-camera path end-to-end: the cube draws against
//! `scene.cameras[0]` (Camera3D), the label draws against `scene.cameras[1]`
//! (Camera2D), and `DepthMode::Disabled` on the HUD material guarantees the
//! glyphs paint on top of the 3D geometry regardless of depth values.
//!
//! Font baking has moved into the engine ([`engine::resources::FontAtlas`]);
//! this file is just the multi-instance per-glyph behaviour + the
//! HUD-textured material registration.

use std::any::Any;

use engine::graphics_manager::{
    Binding, DepthMode, GraphicsManager, MaterialDesc, MaterialHandle, ModelHandle, TextureHandle,
    VertexAttr,
};
use engine::resources::{FontAtlas, Resources};
use engine::scene::Behaviour;
use engine::{Mat4, Vec3};

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

/// Multi-instance label painted with `hud_material` against `atlas.texture`.
/// `scale` converts atlas pixels to HUD world units (the Camera2D in slot 1
/// uses `half_height = 1`, so HUD-world Y ∈ [-1, 1]).
pub struct HudLabelBehaviour {
    // Hold the RAII atlas so its texture outlives every glyph instance below.
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
    }

    fn on_despawn(&mut self, gm: &mut GraphicsManager) {
        for g in &self.glyphs {
            gm.unregister_textured_instance(g.handle);
        }
    }
}
