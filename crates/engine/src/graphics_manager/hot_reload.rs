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

use ash::vk;

use crate::graphics_manager::material::{
    allocate_textured_descriptor_sets, create_pipeline as create_material_pipeline,
};
use crate::graphics_manager::structures::ModelMesh;
use crate::graphics_manager::vulkan;
use crate::graphics_manager::{
    destroy_mesh, destroy_texture_resources, GraphicsManager, MaterialHandle, MeshBuffers,
    MeshHandle, TextureHandle, TextureResources,
};

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

impl GraphicsManager {
    /// Re-upload `png_bytes` into the GPU image backing `handle`, keeping the
    /// handle slot stable so every outstanding instance and every descriptor
    /// set bound against it continues to work. Used by the asset watcher; not
    /// part of the public game-side API today, but `pub` so the watcher (a
    /// crate-internal module) can call it across the `resources` boundary.
    ///
    /// Image dimensions are allowed to change between reloads — the old
    /// image+memory+view are destroyed and re-created; descriptor sets that
    /// referenced the old view are re-bound to the new one.
    ///
    /// Returns `Err(String)` on PNG decode error or unknown handle so the
    /// watcher can log and keep the old texture rather than panicking.
    pub fn reload_texture(
        &mut self,
        handle: TextureHandle,
        png_bytes: &[u8],
    ) -> Result<(), String> {
        if !self.textures.contains_key(&handle) {
            return Err(format!("reload_texture: unknown TextureHandle {handle:?}"));
        }
        // Decode first so a malformed PNG fails *before* we destroy the
        // existing image. Re-implements the relevant bits of
        // `vulkan::load_texture_image` so the early-return on decode error
        // doesn't leak a half-created GPU image.
        let decoded = match image::load_from_memory(png_bytes) {
            Ok(d) => d.to_rgba8(),
            Err(e) => return Err(format!("PNG decode failed: {e}")),
        };
        let (width, height) = (decoded.width(), decoded.height());
        let pixels = decoded.into_raw();

        unsafe {
            self.device
                .device_wait_idle()
                .expect("device_wait_idle failed in reload_texture");
        }

        // Drop descriptor sets pointing at this texture. They'll be re-bound
        // against the new view below.
        let mut materials_to_rebind: Vec<MaterialHandle> = Vec::new();
        for (&mh, mat) in self.materials.iter_mut() {
            if let Some(sets) = mat.texture_descriptor_sets.remove(&handle) {
                if !sets.is_empty() {
                    unsafe {
                        let _ = self
                            .device
                            .free_descriptor_sets(self.descriptor_pool, &sets);
                    }
                }
                materials_to_rebind.push(mh);
            }
        }

        // Tear down old image resources, leaving the registry entry in place
        // (we overwrite it below).
        let old = self
            .textures
            .remove(&handle)
            .expect("reload_texture: texture vanished mid-operation");
        destroy_texture_resources(&self.device, &old);

        let (image, memory) = vulkan::upload_rgba_image(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            width,
            height,
            &pixels,
        );
        let view = vulkan::create_image_view(
            &self.device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        let sampler = vulkan::create_texture_sampler(&self.device);
        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler },
        );

        // Re-bind descriptor sets for every material that was previously
        // sampling this texture. Without this, the next draw against any of
        // those materials would have no descriptor set entry for this handle.
        let new_tex = &self.textures[&handle];
        for mh in materials_to_rebind {
            let mat = self
                .materials
                .get_mut(&mh)
                .expect("reload_texture: material vanished mid-operation");
            let camera_buffers = self
                .camera_uniform_buffers
                .get(&mat.camera_slot)
                .expect("reload_texture: camera slot missing");
            let sets = allocate_textured_descriptor_sets(
                &self.device,
                self.descriptor_pool,
                mat.descriptor_set_layout,
                camera_buffers,
                new_tex.view,
                new_tex.sampler,
            );
            mat.texture_descriptor_sets.insert(handle, sets);
        }
        Ok(())
    }

    /// Re-upload `mesh`'s vertex + index data into the GPU buffers backing
    /// `handle`, keeping the handle stable. Companion to `reload_texture`;
    /// see that method's docs for the watcher use case.
    pub fn reload_mesh(
        &mut self,
        handle: MeshHandle,
        mesh: &ModelMesh,
    ) -> Result<(), String> {
        if !self.meshes.contains_key(&handle) {
            return Err(format!("reload_mesh: unknown MeshHandle {handle:?}"));
        }
        unsafe {
            self.device
                .device_wait_idle()
                .expect("device_wait_idle failed in reload_mesh");
        }
        let old = self
            .meshes
            .remove(&handle)
            .expect("reload_mesh: mesh vanished mid-operation");
        destroy_mesh(&self.device, &old);
        let (vertex_buffer, vertex_memory) = vulkan::create_vertex_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.vertex_bytes,
        );
        let (index_buffer, index_memory) = vulkan::create_index_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.indices,
        );
        self.meshes.insert(
            handle,
            MeshBuffers {
                vertex_buffer,
                vertex_memory,
                index_buffer,
                index_memory,
                index_count: mesh.indices.len() as u32,
            },
        );
        Ok(())
    }

    /// Drain the hot-reload watcher and rebuild any built-in material pipeline
    /// whose SPV bytes changed on disk. Called from `draw_frame` once per
    /// frame. Errors (missing file, IO error, Vulkan pipeline-creation
    /// failure) are logged to stderr and leave the old pipeline running.
    pub(super) fn process_hot_reloads(&mut self) {
        let Some(hr) = self.hot_reload.as_mut() else {
            return;
        };
        let changes = hr.drain();
        if changes.is_empty() {
            return;
        }
        for (handle, stage, path) in changes {
            let new_bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("hot-reload: read {} failed: {e}", path.display());
                    continue;
                }
            };
            let Some(mat) = self.materials.get_mut(&handle) else {
                continue;
            };
            // Stash old SPV so we can roll back on Vulkan failure.
            let backup = match stage {
                ShaderStage::Vertex => {
                    std::mem::replace(&mut mat.vertex_spv, new_bytes)
                }
                ShaderStage::Fragment => {
                    std::mem::replace(&mut mat.fragment_spv, new_bytes)
                }
            };
            // Build new pipeline against the same descriptor-set layout and
            // current swapchain extent. `material::create_pipeline` is the
            // same code path `recreate_swapchain` uses.
            //
            // Wait for the GPU to be idle before destroying the old pipeline:
            // a frame already in flight may still be bound to it.
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in process_hot_reloads");
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_material_pipeline(
                    &self.device,
                    self.render_pass,
                    self.swapchain_extent,
                    mat.descriptor_set_layout,
                    self.pipeline_cache,
                    &mat.vertex_spv,
                    &mat.fragment_spv,
                    &mat.vertex_attrs,
                    &mat.instance_attrs,
                    mat.depth,
                    mat.blend,
                )
            }));
            match result {
                Ok((pipeline, pipeline_layout, _vs, _is)) => {
                    unsafe {
                        self.device.destroy_pipeline(mat.pipeline, None);
                        self.device
                            .destroy_pipeline_layout(mat.pipeline_layout, None);
                    }
                    mat.pipeline = pipeline;
                    mat.pipeline_layout = pipeline_layout;
                    eprintln!("hot-reload: reloaded {}", path.display());
                }
                Err(_) => {
                    // Pipeline creation panicked (e.g. bad SPV header). Roll
                    // back the SPV bytes so the next save can be detected as
                    // a real change.
                    match stage {
                        ShaderStage::Vertex => mat.vertex_spv = backup,
                        ShaderStage::Fragment => mat.fragment_spv = backup,
                    }
                    eprintln!(
                        "hot-reload: pipeline rebuild failed for {} — kept old pipeline",
                        path.display()
                    );
                }
            }
        }
    }
}
