//! Resource management: RAII handles for meshes and textures, plus content
//! loaders (font atlases, OBJ / glTF meshes).
//!
//! Game code talks to [`Resources`] instead of calling the renderer's
//! `register_*` methods directly. Loaders return owning wrappers ([`Mesh`],
//! [`Texture`]) that queue a destroy on `Drop`; [`Resources::flush_pending`]
//! drains that queue against the renderer at a frame boundary, which is when
//! it is safe to call `device_wait_idle` + free.
//!
//! The wrappers expose their inner `MeshHandle` / `TextureHandle` (`Copy` IDs)
//! via `handle()` so callers can still register instances against them through
//! [`GraphicsManager::register_instance`] / `register_textured_instance` —
//! instances are not RAII (see ARCHITECTURE.md Phase 2 for that ownership).
//!
//! Content loaders ([`Resources::load_font`], [`Resources::load_obj`],
//! [`Resources::load_gltf`]) parse asset bytes and produce engine types
//! (RAII handles, [`FontAtlas`], [`MeshData`]). The OBJ + glTF loaders are
//! gated behind the `obj` / `gltf` Cargo features so games that don't need
//! them pay zero dep cost.

pub mod font;
pub mod model;

pub use font::{FontAtlas, GlyphInfo};
pub use model::{pack_lit_vertices, MeshData};

use std::cell::RefCell;
use std::rc::Rc;

use crate::audio::Sound;
use crate::graphics_manager::structures::ModelMesh;
use crate::graphics_manager::{
    GraphicsManager, MaterialDesc, MaterialHandle, MeshHandle, TextureHandle,
};

struct PendingDestroys {
    meshes: Vec<MeshHandle>,
    textures: Vec<TextureHandle>,
}

impl PendingDestroys {
    fn new() -> Self {
        Self {
            meshes: Vec::new(),
            textures: Vec::new(),
        }
    }
}

pub struct Resources {
    pending: Rc<RefCell<PendingDestroys>>,
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}

impl Resources {
    pub fn new() -> Self {
        Self {
            pending: Rc::new(RefCell::new(PendingDestroys::new())),
        }
    }

    pub fn load_mesh(&mut self, gm: &mut GraphicsManager, mesh: &ModelMesh) -> Mesh {
        let handle = gm.register_mesh(mesh);
        Mesh {
            handle,
            pending: Rc::clone(&self.pending),
        }
    }

    pub fn load_texture_png(&mut self, gm: &mut GraphicsManager, bytes: &[u8]) -> Texture {
        let handle = gm.register_texture(bytes);
        Texture {
            handle,
            pending: Rc::clone(&self.pending),
        }
    }

    pub fn load_texture_rgba(
        &mut self,
        gm: &mut GraphicsManager,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Texture {
        let handle = gm.register_texture_rgba(width, height, rgba);
        Texture {
            handle,
            pending: Rc::clone(&self.pending),
        }
    }

    /// Register a custom material (vertex+fragment shaders + vertex/instance
    /// layout + descriptor bindings) with the renderer and return its handle.
    ///
    /// Unlike [`Mesh`] / [`Texture`], materials are not deferred-destroy RAII
    /// today: the returned [`MaterialHandle`] is a plain `Copy` id with no
    /// `Drop`. The renderer owns the material's pipeline + descriptor-set
    /// layout + per-frame instance buffers for the rest of its lifetime.
    /// (Material destruction can be added when a game needs it — e.g. for a
    /// debug shader-reload PR — and would slot into `PendingDestroys` next to
    /// meshes and textures.)
    pub fn load_material(
        &self,
        gm: &mut GraphicsManager,
        desc: &MaterialDesc,
    ) -> MaterialHandle {
        gm.register_material(desc)
    }

    /// Decode `bytes` (any container `kira` supports — mp3 in the default
    /// engine build) into a [`Sound`]. The returned `Sound` is cheap to clone;
    /// hand a clone to every consumer that needs to play it.
    ///
    /// Unlike `load_mesh` / `load_texture_*`, this does not touch the
    /// pending-destroys queue: a `Sound` owns plain heap memory through an
    /// internal `Arc`, freed when the last clone drops. No `GraphicsManager`
    /// is required since audio is independent of the renderer.
    pub fn load_sound(&self, bytes: &[u8]) -> Sound {
        Sound::from_bytes(bytes).expect("Resources::load_sound: failed to decode audio bytes")
    }

    /// Bake a font atlas: rasterise printable ASCII at `px`, shelf-pack into
    /// an RGBA8 texture (white RGB + alpha = bitmap mask, fragment-shader
    /// `discard` handles the masking), upload, and return a [`FontAtlas`]
    /// with the RAII texture + glyph metadata. See [`font::FontAtlas`].
    pub fn load_font(&mut self, gm: &mut GraphicsManager, bytes: &[u8], px: f32) -> FontAtlas {
        FontAtlas::build(self, gm, bytes, px)
    }

    /// Parse a Wavefront OBJ blob into per-attribute [`MeshData`] arrays
    /// (one entry per object/group in the file, in declaration order).
    /// Generates smoothed vertex normals if the file omits them; fills UVs
    /// with `(0, 0)` if absent. Low-level: doesn't touch the renderer. Pair
    /// with [`pack_lit_vertices`] + [`Self::load_mesh`] for a custom vertex
    /// layout, or call [`Self::load_obj`] for the high-level path.
    #[cfg(feature = "obj")]
    pub fn load_obj_data(&self, bytes: &[u8]) -> Vec<MeshData> {
        model::parse_obj(bytes)
    }

    /// Parse OBJ bytes, pack each sub-mesh into the engine's lit vertex
    /// layout (`pos + normal + uv`, stride 32), register against the
    /// renderer, return RAII [`Mesh`] handles. The returned meshes are
    /// ready to draw against any material declaring
    /// `vertex_attrs: [F32x3, F32x3, F32x2]` (e.g. test-3d's lit material).
    #[cfg(feature = "obj")]
    pub fn load_obj(&mut self, gm: &mut GraphicsManager, bytes: &[u8]) -> Vec<Mesh> {
        model::parse_obj(bytes)
            .iter()
            .map(|d| self.load_mesh(gm, &pack_lit_vertices(d)))
            .collect()
    }

    /// Parse a glTF blob into per-primitive [`MeshData`] arrays — one entry
    /// per primitive across all meshes, in declaration order. Generates
    /// smoothed vertex normals if a primitive omits them; fills UVs with
    /// `(0, 0)` if absent. **Only `.glb` and `.gltf` with embedded data URI
    /// buffers are supported** — `.gltf` with external `.bin` files panics
    /// with a clear message (the loader has no filesystem context to
    /// resolve URIs).
    #[cfg(feature = "gltf")]
    pub fn load_gltf_data(&self, bytes: &[u8]) -> Vec<MeshData> {
        model::parse_gltf(bytes)
    }

    /// High-level glTF loader: parse, pack into the lit vertex layout,
    /// register, return RAII [`Mesh`] handles. See [`Self::load_gltf_data`]
    /// for the supported-format caveats.
    #[cfg(feature = "gltf")]
    pub fn load_gltf(&mut self, gm: &mut GraphicsManager, bytes: &[u8]) -> Vec<Mesh> {
        model::parse_gltf(bytes)
            .iter()
            .map(|d| self.load_mesh(gm, &pack_lit_vertices(d)))
            .collect()
    }

    /// Drain any queued resource destroys. Must be called at a point where the
    /// GPU is not actively reading the resources — currently between frames,
    /// before `draw_frame`. Each `unregister_*` issues its own `device_wait_idle`.
    pub fn flush_pending(&mut self, gm: &mut GraphicsManager) {
        let (meshes, textures) = {
            let mut pending = self.pending.borrow_mut();
            if pending.meshes.is_empty() && pending.textures.is_empty() {
                return;
            }
            (
                std::mem::take(&mut pending.meshes),
                std::mem::take(&mut pending.textures),
            )
        };
        for h in meshes {
            gm.unregister_mesh(h);
        }
        for h in textures {
            gm.unregister_texture(h);
        }
    }
}

pub struct Mesh {
    handle: MeshHandle,
    pending: Rc<RefCell<PendingDestroys>>,
}

impl Mesh {
    pub fn handle(&self) -> MeshHandle {
        self.handle
    }
}

impl Drop for Mesh {
    fn drop(&mut self) {
        self.pending.borrow_mut().meshes.push(self.handle);
    }
}

pub struct Texture {
    handle: TextureHandle,
    pending: Rc<RefCell<PendingDestroys>>,
}

impl Texture {
    pub fn handle(&self) -> TextureHandle {
        self.handle
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        self.pending.borrow_mut().textures.push(self.handle);
    }
}
