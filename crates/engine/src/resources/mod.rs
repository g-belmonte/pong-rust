//! Resource management: refcounted shared handles for meshes and textures,
//! plus content loaders (font atlases, OBJ / glTF meshes).
//!
//! Game code talks to [`Resources`] instead of calling the renderer's
//! `register_*` methods directly. Loaders return clonable handles ([`Mesh`],
//! [`Texture`]) that wrap an internal `Rc`; the GPU resource is destroyed
//! when the last clone drops. Drops queue into a pending-destroy list which
//! [`Resources::flush_pending`] drains at a frame boundary, where it is safe
//! to call `device_wait_idle` + free.
//!
//! ## Sharing across scenes
//!
//! [`Resources`] keeps a content-addressed cache keyed by a hash of the input
//! bytes. `load_mesh` / `load_texture_*` look up the cache first: if a live
//! clone is still alive somewhere (refcount ≥ 1) they return another clone
//! against the existing GPU resource; otherwise they upload fresh and store a
//! [`std::rc::Weak`] in the cache. This means two scenes that load the same
//! assets share the same GPU memory, and a scene switch that retains shared
//! assets in its new behaviours never re-uploads them.
//!
//! Combine with [`UpdateCtx::request_scene`](crate::scene::UpdateCtx::request_scene)
//! and the App's swap logic (new builder runs *before* old scene is torn
//! down, so shared assets keep refcount ≥ 1 across the transition) to get
//! "destroy what isn't reused, keep what is, upload what's new" behaviour
//! implicitly.
//!
//! The cache holds `Weak`s, so a cached entry never keeps an asset alive on
//! its own — the asset survives only as long as game code keeps at least one
//! [`Mesh`] / [`Texture`] clone. Stale `Weak`s are garbage-collected lazily
//! inside `flush_pending`.
//!
//! ## Handle identity
//!
//! [`Mesh::handle`] / [`Texture::handle`] still expose the renderer's `Copy`
//! ID for callers that need to register instances against them through
//! [`GraphicsManager::register_instance`] / `register_textured_instance` —
//! instances are not RAII (see ARCHITECTURE.md Phase 2 for that ownership).
//! Two clones of the same `Mesh` return the same handle.

#[cfg(feature = "hot-reload")]
pub mod asset_watcher;
pub mod font;
pub mod model;

pub use font::{FontAtlas, GlyphInfo};
pub use model::{pack_lit_vertices, MeshData};

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
#[cfg(feature = "hot-reload")]
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use crate::audio::Sound;
use crate::graphics_manager::structures::ModelMesh;
use crate::graphics_manager::{
    GraphicsManager, MaterialDesc, MaterialHandle, MeshHandle, TextureHandle,
};

/// A reference to an asset payload, optionally tagged with its source path.
///
/// Built by the [`crate::asset!`] macro: in release the macro expands to
/// `AssetSource::from_bytes(include_bytes!(...))` so the bytes are baked into
/// the binary and the `path` field is `None`; in dev (with the `hot-reload`
/// feature) the macro expands to `AssetSource::from_file(...)` which reads
/// the bytes off disk at runtime and remembers the absolute path so the
/// asset watcher can re-run loaders when the file changes.
///
/// Direct construction via [`Self::from_bytes`] is also fine for callers that
/// just want to load an in-memory buffer (no hot-reload tracking).
pub struct AssetSource<'a> {
    bytes: Cow<'a, [u8]>,
    #[cfg(feature = "hot-reload")]
    path: Option<PathBuf>,
}

impl<'a> AssetSource<'a> {
    /// Borrow an existing byte slice. Always available, both in release and
    /// dev builds. No hot-reload tracking — pair with [`Self::from_file`]
    /// (dev only) if you need watcher integration.
    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        Self {
            bytes: Cow::Borrowed(bytes),
            #[cfg(feature = "hot-reload")]
            path: None,
        }
    }

    /// Read `path` from disk and remember it for the asset watcher. Panics
    /// if the read fails — every `asset!` call site is a developer-supplied
    /// path that should exist; a typo should fail loud at startup.
    #[cfg(feature = "hot-reload")]
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
        let path = path.into();
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("AssetSource::from_file: read {} failed: {e}", path.display()));
        Self {
            bytes: Cow::Owned(bytes),
            path: Some(path),
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    #[cfg(feature = "hot-reload")]
    pub fn path(&self) -> Option<&std::path::Path> {
        self.path.as_deref()
    }
}

impl<'a> From<&'a [u8]> for AssetSource<'a> {
    fn from(b: &'a [u8]) -> Self {
        Self::from_bytes(b)
    }
}

/// Build an [`AssetSource`] from a path **relative to the calling crate's
/// `Cargo.toml`**. In release builds the asset bytes are baked into the
/// binary via `include_bytes!`; in dev builds (with the `hot-reload` feature
/// enabled) the bytes are read from disk and the file is registered with the
/// asset watcher so any later edit re-runs the loader.
///
/// ```ignore
/// // pong/src/ball.rs:
/// let tex = resources.load_texture_png(gm, engine::asset!("assets/tennis-ball.png"));
/// ```
///
/// The `CARGO_MANIFEST_DIR` env-var is resolved at the expansion site, so
/// `asset!` correctly anchors at the *game* crate's root even though the
/// macro lives in the engine crate.
#[cfg(not(feature = "hot-reload"))]
#[macro_export]
macro_rules! asset {
    ($path:literal) => {
        $crate::resources::AssetSource::from_bytes(::std::include_bytes!(::std::concat!(
            ::std::env!("CARGO_MANIFEST_DIR"),
            "/",
            $path
        )))
    };
}

#[cfg(feature = "hot-reload")]
#[macro_export]
macro_rules! asset {
    ($path:literal) => {
        $crate::resources::AssetSource::from_file(::std::concat!(
            ::std::env!("CARGO_MANIFEST_DIR"),
            "/",
            $path
        ))
    };
}

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
    // Content-addressed caches. Keys are 64-bit hashes of the input bytes
    // (plus dimensions where relevant). Values are `Weak`s so the cache does
    // not keep assets alive — the asset survives only as long as game code
    // holds at least one `Mesh` / `Texture` clone.
    cache_meshes: HashMap<u64, Weak<MeshInner>>,
    cache_textures: HashMap<u64, Weak<TextureInner>>,
    #[cfg(feature = "hot-reload")]
    asset_watcher: Option<asset_watcher::AssetWatcher>,
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
            cache_meshes: HashMap::new(),
            cache_textures: HashMap::new(),
            #[cfg(feature = "hot-reload")]
            asset_watcher: asset_watcher::AssetWatcher::try_new(),
        }
    }

    pub fn load_mesh(&mut self, gm: &mut GraphicsManager, mesh: &ModelMesh) -> Mesh {
        let key = hash_mesh(mesh);
        if let Some(existing) = self.cache_meshes.get(&key).and_then(|w| w.upgrade()) {
            return Mesh { inner: existing };
        }
        let handle = gm.register_mesh(mesh);
        let inner = Rc::new(MeshInner {
            handle,
            pending: Rc::clone(&self.pending),
        });
        self.cache_meshes.insert(key, Rc::downgrade(&inner));
        Mesh { inner }
    }

    pub fn load_texture_png(
        &mut self,
        gm: &mut GraphicsManager,
        src: AssetSource<'_>,
    ) -> Texture {
        let bytes = src.bytes();
        let key = hash_bytes_tagged(b"png", bytes);
        let texture = if let Some(existing) = self.cache_textures.get(&key).and_then(|w| w.upgrade()) {
            Texture { inner: existing }
        } else {
            let handle = gm.register_texture(bytes);
            let inner = Rc::new(TextureInner {
                handle,
                pending: Rc::clone(&self.pending),
            });
            self.cache_textures.insert(key, Rc::downgrade(&inner));
            Texture { inner }
        };
        #[cfg(feature = "hot-reload")]
        if let (Some(path), Some(watcher)) = (src.path(), self.asset_watcher.as_mut()) {
            watcher.register_texture_png(
                path,
                Rc::downgrade(&texture.inner),
                texture.inner.handle,
            );
        }
        texture
    }

    pub fn load_texture_rgba(
        &mut self,
        gm: &mut GraphicsManager,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Texture {
        let key = hash_rgba(width, height, rgba);
        if let Some(existing) = self.cache_textures.get(&key).and_then(|w| w.upgrade()) {
            return Texture { inner: existing };
        }
        let handle = gm.register_texture_rgba(width, height, rgba);
        let inner = Rc::new(TextureInner {
            handle,
            pending: Rc::clone(&self.pending),
        });
        self.cache_textures.insert(key, Rc::downgrade(&inner));
        Texture { inner }
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

    /// Decode `src`'s bytes (any container `kira` supports — mp3 in the
    /// default engine build) into a [`Sound`]. The returned `Sound` is cheap
    /// to clone; hand a clone to every consumer that needs to play it.
    ///
    /// Unlike `load_mesh` / `load_texture_*`, this does not touch the
    /// pending-destroys queue: a `Sound` owns plain heap memory through an
    /// internal refcounted cell, freed when the last clone drops. No
    /// `GraphicsManager` is required since audio is independent of the
    /// renderer.
    ///
    /// In dev (`hot-reload` feature on) the source file is registered with
    /// the asset watcher; a later edit re-decodes the file and swaps the
    /// data inside this `Sound` (and every clone of it).
    pub fn load_sound(&mut self, src: AssetSource<'_>) -> Sound {
        let sound = Sound::from_bytes(src.bytes())
            .expect("Resources::load_sound: failed to decode audio bytes");
        #[cfg(feature = "hot-reload")]
        if let (Some(path), Some(watcher)) = (src.path(), self.asset_watcher.as_mut()) {
            watcher.register_sound(path, sound.clone());
        }
        sound
    }

    /// Bake a font atlas: rasterise printable ASCII at `px`, shelf-pack into
    /// an RGBA8 texture (white RGB + alpha = bitmap mask, fragment-shader
    /// `discard` handles the masking), upload, and return a [`FontAtlas`]
    /// with the RAII texture + glyph metadata. See [`font::FontAtlas`].
    ///
    /// The baked atlas bytes go through [`load_texture_rgba`](Self::load_texture_rgba),
    /// so the GPU texture is shared across two `load_font` calls with the
    /// same `(bytes, px)`. The CPU bake (rasterise + shelf-pack) does run
    /// again on each call — sub-millisecond for ASCII at typical sizes.
    pub fn load_font(
        &mut self,
        gm: &mut GraphicsManager,
        src: AssetSource<'_>,
        px: f32,
    ) -> FontAtlas {
        let atlas = FontAtlas::build(self, gm, src.bytes(), px);
        // Font hot-reload is log-only — re-baking can change glyph metrics,
        // which would leave already-laid-out text positioned against stale
        // `GlyphInfo`. Register so the watcher prints a "restart to apply"
        // line; don't try to swap atlas bytes.
        #[cfg(feature = "hot-reload")]
        if let (Some(path), Some(watcher)) = (src.path(), self.asset_watcher.as_mut()) {
            watcher.register_font(path);
        }
        atlas
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
    pub fn load_obj(&mut self, gm: &mut GraphicsManager, src: AssetSource<'_>) -> Vec<Mesh> {
        let meshes: Vec<Mesh> = model::parse_obj(src.bytes())
            .iter()
            .map(|d| self.load_mesh(gm, &pack_lit_vertices(d)))
            .collect();
        #[cfg(feature = "hot-reload")]
        if let (Some(path), Some(watcher)) = (src.path(), self.asset_watcher.as_mut()) {
            let entries: Vec<asset_watcher::MeshEntry> = meshes
                .iter()
                .map(|m| asset_watcher::MeshEntry {
                    weak: Rc::downgrade(&m.inner),
                    handle: m.inner.handle,
                })
                .collect();
            watcher.register_obj(path, entries);
        }
        meshes
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
    pub fn load_gltf(&mut self, gm: &mut GraphicsManager, src: AssetSource<'_>) -> Vec<Mesh> {
        let meshes: Vec<Mesh> = model::parse_gltf(src.bytes())
            .iter()
            .map(|d| self.load_mesh(gm, &pack_lit_vertices(d)))
            .collect();
        #[cfg(feature = "hot-reload")]
        if let (Some(path), Some(watcher)) = (src.path(), self.asset_watcher.as_mut()) {
            let entries: Vec<asset_watcher::MeshEntry> = meshes
                .iter()
                .map(|m| asset_watcher::MeshEntry {
                    weak: Rc::downgrade(&m.inner),
                    handle: m.inner.handle,
                })
                .collect();
            watcher.register_gltf(path, entries);
        }
        meshes
    }

    /// Drain any queued resource destroys and garbage-collect stale cache
    /// entries (`Weak`s whose target has dropped). Must be called at a point
    /// where the GPU is not actively reading the resources — currently
    /// between frames, before `draw_frame`. Each `unregister_*` issues its
    /// own `device_wait_idle`.
    pub fn flush_pending(&mut self, gm: &mut GraphicsManager) {
        let (meshes, textures) = {
            let mut pending = self.pending.borrow_mut();
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
        // GC stale cache entries. Cheap; runs once per frame against a small
        // map (one entry per distinct loaded asset).
        self.cache_meshes.retain(|_, w| w.strong_count() > 0);
        self.cache_textures.retain(|_, w| w.strong_count() > 0);
        #[cfg(feature = "hot-reload")]
        if let Some(watcher) = self.asset_watcher.as_mut() {
            watcher.gc();
        }
    }

    /// Drain queued asset-file events and apply reloads. Called from `App`
    /// once per frame, right before `flush_pending`. No-op when the
    /// `hot-reload` Cargo feature is off — the `App` still calls it, but the
    /// body compiles to nothing.
    #[cfg(feature = "hot-reload")]
    pub fn process_hot_reloads(&mut self, gm: &mut GraphicsManager) {
        if let Some(watcher) = self.asset_watcher.as_mut() {
            watcher.poll_and_apply(gm);
        }
    }

    #[cfg(not(feature = "hot-reload"))]
    pub fn process_hot_reloads(&mut self, _gm: &mut GraphicsManager) {}
}

/// Inner refcounted body of a [`Mesh`]. `Drop` queues GPU destruction for
/// the next [`Resources::flush_pending`] — only fires when the last [`Mesh`]
/// clone is dropped.
pub(crate) struct MeshInner {
    pub(crate) handle: MeshHandle,
    pending: Rc<RefCell<PendingDestroys>>,
}

impl Drop for MeshInner {
    fn drop(&mut self) {
        self.pending.borrow_mut().meshes.push(self.handle);
    }
}

/// Refcounted handle to a registered mesh. Cheap to clone (one `Rc::clone`);
/// every clone shares the same underlying GPU buffer and the same renderer
/// [`MeshHandle`]. The GPU buffer is destroyed when the last clone drops.
#[derive(Clone)]
pub struct Mesh {
    inner: Rc<MeshInner>,
}

impl Mesh {
    pub fn handle(&self) -> MeshHandle {
        self.inner.handle
    }
}

/// Inner refcounted body of a [`Texture`]. See [`MeshInner`].
pub(crate) struct TextureInner {
    pub(crate) handle: TextureHandle,
    pending: Rc<RefCell<PendingDestroys>>,
}

impl Drop for TextureInner {
    fn drop(&mut self) {
        self.pending.borrow_mut().textures.push(self.handle);
    }
}

/// Refcounted handle to a registered texture. Cheap to clone; every clone
/// shares the same underlying GPU image and the same renderer
/// [`TextureHandle`]. The GPU image is destroyed when the last clone drops.
#[derive(Clone)]
pub struct Texture {
    inner: Rc<TextureInner>,
}

impl Texture {
    pub fn handle(&self) -> TextureHandle {
        self.inner.handle
    }
}

// --- hashing helpers ---------------------------------------------------------

fn hash_mesh(mesh: &ModelMesh) -> u64 {
    let mut h = DefaultHasher::new();
    b"mesh".hash(&mut h);
    mesh.vertex_stride.hash(&mut h);
    mesh.vertex_bytes.hash(&mut h);
    mesh.indices.hash(&mut h);
    h.finish()
}

fn hash_bytes_tagged(tag: &[u8], bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    tag.hash(&mut h);
    bytes.hash(&mut h);
    h.finish()
}

fn hash_rgba(width: u32, height: u32, bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    b"rgba".hash(&mut h);
    width.hash(&mut h);
    height.hash(&mut h);
    bytes.hash(&mut h);
    h.finish()
}
