//! Resource management: RAII handles for meshes and textures.
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

use std::cell::RefCell;
use std::rc::Rc;

use crate::audio::Sound;
use crate::graphics_manager::structures::ModelMesh;
use crate::graphics_manager::{GraphicsManager, MeshHandle, TextureHandle};

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
