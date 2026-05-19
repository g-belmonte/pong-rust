//! Asset hot-reload (non-shader path).
//!
//! Companion to `graphics_manager::hot_reload` (which watches SPV files and
//! rebuilds material pipelines). This module watches the *game-side* asset
//! files that flow through [`crate::resources::Resources`]: textures (PNG),
//! meshes (OBJ/glTF), sounds (MP3), and fonts. It maps each watched file to
//! a [`Weak`] of the asset's inner refcounted handle, so a path's reload
//! work is naturally short-circuited the moment no game code holds a clone
//! of that asset anymore (the upgrade returns `None` and the entry GCs out).
//!
//! Compiled in only when the `hot-reload` Cargo feature is enabled. Failure
//! modes are non-fatal — a missing notify backend, an IO error reading the
//! changed file, or a decode/upload error all log to stderr and leave the
//! previously-loaded asset in place. The dev loop keeps going.
//!
//! Per-asset reload semantics:
//!
//! - **Texture (PNG)**: re-decode, in-place [`GraphicsManager::reload_texture`]
//!   on the same [`TextureHandle`] slot. All sampler descriptor sets that
//!   reference the texture are rebound to the new image view.
//! - **Mesh (OBJ/glTF)**: re-parse + re-pack, in-place
//!   [`GraphicsManager::reload_mesh`] on each affected [`MeshHandle`]. If the
//!   sub-mesh count changes the per-slot update is skipped for the extra/
//!   missing sub-meshes and a warning is logged (restart required).
//! - **Sound (MP3)**: re-decode, swap the data inside the existing
//!   [`crate::audio::Sound`]'s inner cell — every clone of that sound (held
//!   by behaviours that play it) sees the new sample on the next `play` call.
//! - **Font (TTF)**: log-only. Re-baking the atlas can shift glyph metrics,
//!   which would leave previously-laid-out text labels positioned against
//!   stale `GlyphInfo`. Restart to apply.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Weak;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use notify::{recommended_watcher, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

#[cfg(any(feature = "obj", feature = "gltf"))]
use crate::graphics_manager::MeshHandle;
use crate::graphics_manager::{GraphicsManager, TextureHandle};

/// What kind of reload work an event on a watched path should trigger.
pub(crate) enum AssetKind {
    /// PNG bytes → [`GraphicsManager::reload_texture`] on the named handle.
    /// The [`Weak`] gates the work: if no game code holds a clone of this
    /// texture any more, the upgrade returns `None` and we skip silently.
    TexturePng {
        weak: Weak<super::TextureInner>,
        handle: TextureHandle,
    },
    /// OBJ bytes → re-parse, re-pack into the engine's lit vertex layout
    /// (`pos + normal + uv`, stride 32), in-place
    /// [`GraphicsManager::reload_mesh`] for every sub-mesh handle in order.
    #[cfg(feature = "obj")]
    MeshObj { entries: Vec<MeshEntry> },
    /// glTF bytes → same shape as `MeshObj` but routes through the glTF parser.
    #[cfg(feature = "gltf")]
    MeshGltf { entries: Vec<MeshEntry> },
    /// MP3 / other kira-supported audio bytes → decode + swap the inner cell
    /// of every still-alive [`crate::audio::Sound`] clone registered against
    /// this path (one path can map to many sounds when game code re-loads
    /// the same file separately, as pong does with `hit.mp3`).
    Sound { weaks: Vec<crate::audio::Sound> },
    /// TTF bytes → just log; the atlas isn't re-baked because glyph metrics
    /// can shift and stale layouts elsewhere would render at wrong UVs.
    Font,
}

#[cfg(any(feature = "obj", feature = "gltf"))]
pub(crate) struct MeshEntry {
    pub weak: Weak<super::MeshInner>,
    pub handle: MeshHandle,
}

const DEBOUNCE: Duration = Duration::from_millis(150);

pub(crate) struct AssetWatcher {
    // Drop = stop the background thread, so we keep the watcher alive.
    _watcher: RecommendedWatcher,
    rx: mpsc::Receiver<notify::Result<Event>>,
    /// Set of directories already passed to `watcher.watch`. notify's recursive
    /// flag would simplify this but might pick up too much in deeply-nested
    /// trees; non-recursive per parent dir is enough for our `assets/` layout.
    watched_dirs: std::collections::HashSet<PathBuf>,
    /// Canonical absolute path → reload descriptor.
    by_path: HashMap<PathBuf, AssetKind>,
    /// Debounce repeated events for the same path (most editors emit several
    /// notify events per save: atomic-write, chmod, etc.).
    last_seen: HashMap<PathBuf, Instant>,
}

impl AssetWatcher {
    pub(crate) fn try_new() -> Option<Self> {
        let (tx, rx) = mpsc::channel();
        let watcher = match recommended_watcher(tx) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("asset-watcher: failed to create watcher: {e}");
                return None;
            }
        };
        Some(Self {
            _watcher: watcher,
            rx,
            watched_dirs: std::collections::HashSet::new(),
            by_path: HashMap::new(),
            last_seen: HashMap::new(),
        })
    }

    /// Add `path`'s parent directory to the watch set (idempotent) and store
    /// the reload descriptor under the canonical path.
    fn ensure_watching(&mut self, path: &Path) -> Option<PathBuf> {
        let canon = match std::fs::canonicalize(path) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("asset-watcher: canonicalize {} failed: {e}", path.display());
                return None;
            }
        };
        let parent = match canon.parent() {
            Some(p) => p.to_path_buf(),
            None => return Some(canon),
        };
        if !self.watched_dirs.contains(&parent) {
            if let Err(e) = self._watcher.watch(&parent, RecursiveMode::NonRecursive) {
                eprintln!("asset-watcher: failed to watch {}: {e}", parent.display());
                return None;
            }
            eprintln!("asset-watcher: watching {}", parent.display());
            self.watched_dirs.insert(parent);
        }
        Some(canon)
    }

    pub(crate) fn register_texture_png(
        &mut self,
        path: &Path,
        weak: Weak<super::TextureInner>,
        handle: TextureHandle,
    ) {
        let Some(canon) = self.ensure_watching(path) else { return };
        self.by_path.insert(canon, AssetKind::TexturePng { weak, handle });
    }

    #[cfg(feature = "obj")]
    pub(crate) fn register_obj(&mut self, path: &Path, entries: Vec<MeshEntry>) {
        let Some(canon) = self.ensure_watching(path) else { return };
        self.by_path.insert(canon, AssetKind::MeshObj { entries });
    }

    #[cfg(feature = "gltf")]
    pub(crate) fn register_gltf(&mut self, path: &Path, entries: Vec<MeshEntry>) {
        let Some(canon) = self.ensure_watching(path) else { return };
        self.by_path.insert(canon, AssetKind::MeshGltf { entries });
    }

    pub(crate) fn register_sound(&mut self, path: &Path, sound: crate::audio::Sound) {
        let Some(canon) = self.ensure_watching(path) else { return };
        // Multiple game-side `load_sound` calls with the same path produce
        // distinct `Sound`s. They all need their inner data swapped on
        // reload — accumulate clones rather than overwriting.
        match self.by_path.get_mut(&canon) {
            Some(AssetKind::Sound { weaks }) => {
                if !weaks.iter().any(|s| s.id() == sound.id()) {
                    weaks.push(sound);
                }
            }
            _ => {
                self.by_path.insert(canon, AssetKind::Sound { weaks: vec![sound] });
            }
        }
    }

    pub(crate) fn register_font(&mut self, path: &Path) {
        let Some(canon) = self.ensure_watching(path) else { return };
        // Font is log-only; storing the entry just tags the path so the
        // event-drain code knows what message to print.
        self.by_path.entry(canon).or_insert(AssetKind::Font);
    }

    /// Drain queued file events and apply reloads. Called once per frame from
    /// [`crate::resources::Resources::process_hot_reloads`].
    pub(crate) fn poll_and_apply(&mut self, gm: &mut GraphicsManager) {
        let mut changed: Vec<PathBuf> = Vec::new();
        let now = Instant::now();
        while let Ok(res) = self.rx.try_recv() {
            let event = match res {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("asset-watcher: watcher error: {e}");
                    continue;
                }
            };
            match event.kind {
                EventKind::Modify(_) | EventKind::Create(_) => {}
                _ => continue,
            }
            for path in event.paths {
                let canon = std::fs::canonicalize(&path).unwrap_or(path);
                if !self.by_path.contains_key(&canon) {
                    continue;
                }
                if let Some(&last) = self.last_seen.get(&canon) {
                    if now.duration_since(last) < DEBOUNCE {
                        continue;
                    }
                }
                self.last_seen.insert(canon.clone(), now);
                if !changed.contains(&canon) {
                    changed.push(canon);
                }
            }
        }
        for canon in changed {
            self.apply(&canon, gm);
        }
    }

    fn apply(&mut self, path: &Path, gm: &mut GraphicsManager) {
        let kind = match self.by_path.get_mut(path) {
            Some(k) => k,
            None => return,
        };
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("asset-watcher: read {} failed: {e}", path.display());
                return;
            }
        };
        match kind {
            AssetKind::TexturePng { weak, handle } => {
                if weak.upgrade().is_none() {
                    return;
                }
                if let Err(e) = gm.reload_texture(*handle, &bytes) {
                    eprintln!(
                        "asset-watcher: reload_texture {} failed: {e}",
                        path.display()
                    );
                } else {
                    eprintln!("asset-watcher: reloaded texture {}", path.display());
                }
            }
            #[cfg(feature = "obj")]
            AssetKind::MeshObj { entries } => {
                let meshes = super::model::parse_obj(&bytes);
                apply_mesh_reload(gm, entries, &meshes, path, "OBJ");
            }
            #[cfg(feature = "gltf")]
            AssetKind::MeshGltf { entries } => {
                let meshes = super::model::parse_gltf(&bytes);
                apply_mesh_reload(gm, entries, &meshes, path, "glTF");
            }
            AssetKind::Sound { weaks } => {
                // GC dead clones first; they correspond to game code that
                // dropped the sound.
                let mut updated = 0;
                for s in weaks.iter() {
                    if let Err(e) = s.replace_from_bytes(&bytes) {
                        eprintln!(
                            "asset-watcher: sound decode {} failed: {e:?}",
                            path.display()
                        );
                        return;
                    }
                    updated += 1;
                }
                eprintln!(
                    "asset-watcher: reloaded sound {} ({} clones)",
                    path.display(),
                    updated
                );
            }
            AssetKind::Font => {
                eprintln!(
                    "asset-watcher: font {} changed — restart to apply (glyph metrics may shift)",
                    path.display()
                );
            }
        }
    }

    /// GC entries whose target asset has been dropped by game code.
    pub(crate) fn gc(&mut self) {
        self.by_path.retain(|_, kind| match kind {
            AssetKind::TexturePng { weak, .. } => weak.strong_count() > 0,
            #[cfg(feature = "obj")]
            AssetKind::MeshObj { entries } => entries.iter().any(|e| e.weak.strong_count() > 0),
            #[cfg(feature = "gltf")]
            AssetKind::MeshGltf { entries } => entries.iter().any(|e| e.weak.strong_count() > 0),
            AssetKind::Sound { weaks } => {
                // Drop sounds whose only ref left is the watcher's own clone
                // (strong_count == 1 means no game-side holder remains).
                weaks.retain(|s| s.strong_count() > 1);
                !weaks.is_empty()
            }
            AssetKind::Font => true,
        });
    }
}

#[cfg(any(feature = "obj", feature = "gltf"))]
fn apply_mesh_reload(
    gm: &mut GraphicsManager,
    entries: &[MeshEntry],
    parsed: &[super::MeshData],
    path: &Path,
    kind_label: &str,
) {
    if parsed.len() != entries.len() {
        eprintln!(
            "asset-watcher: {} {} sub-mesh count changed ({} -> {}) — restart to apply",
            kind_label,
            path.display(),
            entries.len(),
            parsed.len()
        );
    }
    let pair_count = entries.len().min(parsed.len());
    let mut reloaded = 0;
    for i in 0..pair_count {
        let entry = &entries[i];
        if entry.weak.upgrade().is_none() {
            continue;
        }
        let packed = super::model::pack_lit_vertices(&parsed[i]);
        if let Err(e) = gm.reload_mesh(entry.handle, &packed) {
            eprintln!(
                "asset-watcher: reload_mesh {} sub-mesh {i} failed: {e}",
                path.display()
            );
            continue;
        }
        reloaded += 1;
    }
    eprintln!(
        "asset-watcher: reloaded {} {} ({} sub-meshes)",
        kind_label,
        path.display(),
        reloaded
    );
}

