//! Shader hot-reload (Phase 7 Part 2).
//!
//! Compiled in only when the `hot-reload` Cargo feature is enabled. Watches
//! `crates/engine/shaders/spv/` (the SPIR-V output directory; you still run
//! `scripts/compile-shaders.sh` to regenerate the SPV bytes) and signals which
//! [`MaterialHandle`]'s [`ShaderStage`] needs rebuilding. The actual pipeline
//! rebuild lives in `GraphicsManager::process_hot_reloads`.
//!
//! Failure modes are non-fatal by design — a missing watch dir, a broken
//! `notify` backend, an IO error reading the new SPV, or a Vulkan failure
//! building the new pipeline all log to stderr and leave the old pipeline
//! running. The dev loop keeps going so the user can fix the error and save
//! again.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use notify::{recommended_watcher, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

use crate::graphics_manager::MaterialHandle;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShaderStage {
    Vertex,
    Fragment,
}

/// Watches an SPV directory and records which (material, stage) each watched
/// file maps to. Owned by [`crate::graphics_manager::GraphicsManager`].
pub(crate) struct HotReload {
    // Keeping the watcher alive is what keeps file events flowing; it's
    // dropped when the struct is dropped, which stops the background thread.
    _watcher: RecommendedWatcher,
    rx: mpsc::Receiver<notify::Result<Event>>,
    /// Canonical absolute path → (material, stage).
    file_to_material: HashMap<PathBuf, (MaterialHandle, ShaderStage)>,
    /// Debounce: ignore events on a path that fired in the last DEBOUNCE.
    /// Many editors emit several events per save (atomic write + chmod, etc.);
    /// without this we'd rebuild the pipeline two or three times per save.
    last_seen: HashMap<PathBuf, Instant>,
}

const DEBOUNCE: Duration = Duration::from_millis(150);

impl HotReload {
    /// Try to start a watcher on `watch_dir`. Returns `None` (with a logged
    /// reason) when watching can't be set up — e.g. the directory doesn't
    /// exist yet or the platform's notify backend failed.
    pub(crate) fn try_new(watch_dir: &Path) -> Option<Self> {
        let (tx, rx) = mpsc::channel();
        let mut watcher = match recommended_watcher(tx) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("hot-reload: failed to create watcher: {e}");
                return None;
            }
        };
        if let Err(e) = watcher.watch(watch_dir, RecursiveMode::NonRecursive) {
            eprintln!(
                "hot-reload: failed to watch {}: {e}",
                watch_dir.display()
            );
            return None;
        }
        eprintln!("hot-reload: watching {}", watch_dir.display());
        Some(Self {
            _watcher: watcher,
            rx,
            file_to_material: HashMap::new(),
            last_seen: HashMap::new(),
        })
    }

    /// Associate a watched file with the (material, stage) it feeds. Called
    /// once per shader at startup. The path is canonicalised so that the
    /// `notify` events (which the OS reports as absolute paths) can be looked
    /// up reliably.
    pub(crate) fn register(
        &mut self,
        path: &Path,
        material: MaterialHandle,
        stage: ShaderStage,
    ) {
        let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.file_to_material.insert(canon, (material, stage));
    }

    /// Drain queued file events and return the set of (material, stage)
    /// rebuilds the caller should perform. Deduplicates per file (so a single
    /// save that triggers two events only rebuilds once) and applies the
    /// debounce window.
    pub(crate) fn drain(&mut self) -> Vec<(MaterialHandle, ShaderStage, PathBuf)> {
        let mut out: Vec<(MaterialHandle, ShaderStage, PathBuf)> = Vec::new();
        let mut seen_this_call: std::collections::HashSet<PathBuf> =
            std::collections::HashSet::new();
        let now = Instant::now();

        while let Ok(res) = self.rx.try_recv() {
            let event = match res {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("hot-reload: watcher error: {e}");
                    continue;
                }
            };
            // Only care about content-changing events. notify's
            // `EventKind::Any` and `Access` variants don't indicate a write.
            match event.kind {
                EventKind::Modify(_) | EventKind::Create(_) => {}
                _ => continue,
            }
            for path in event.paths {
                let canon =
                    std::fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
                let Some(&(material, stage)) = self.file_to_material.get(&canon) else {
                    continue;
                };
                if seen_this_call.contains(&canon) {
                    continue;
                }
                if let Some(&last) = self.last_seen.get(&canon) {
                    if now.duration_since(last) < DEBOUNCE {
                        continue;
                    }
                }
                self.last_seen.insert(canon.clone(), now);
                seen_this_call.insert(canon.clone());
                out.push((material, stage, canon));
            }
        }
        out
    }
}

/// Compile-time path to `crates/engine/shaders/spv/`. Resolves via
/// `CARGO_MANIFEST_DIR`, which is the absolute path of the engine crate at
/// build time — i.e. exactly where the developer is editing shaders.
pub(crate) fn engine_spv_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders/spv")
}
