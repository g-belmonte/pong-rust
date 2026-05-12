//! Scene + Behaviour API.
//!
//! A flat [`Scene`] of [`Object`]s. Each `Object` carries a [`Transform`],
//! optionally a [`Renderable`] (one solid-coloured or one textured instance),
//! and a `Vec<Box<dyn Behaviour>>`. Game logic lives in behaviours.
//!
//! ## Dispatch model
//!
//! On every frame, the engine walks all objects and calls
//! [`Behaviour::update`] on each behaviour, then again for any incoming
//! [`Event`]s via [`Behaviour::on_event`]. The behaviour being called is
//! temporarily moved out of its parent `Object` for the duration of the call,
//! so a behaviour does not see *itself* in `ctx.scene` during its own
//! `update` — but it can read and mutate everything else, including its own
//! `Object`'s `transform` and `visible` flag via `ctx.scene.get_mut(ctx.self_id)`.
//!
//! ## Spawn / despawn
//!
//! [`Scene::spawn`] and [`Scene::despawn`] are **deferred**: they queue
//! commands and return immediately (spawn pre-allocates an [`ObjectId`] so
//! callers can wire IDs together at build time). [`Scene::apply_commands`]
//! drains the queue against the renderer at a frame boundary (registering
//! instances on spawn, unregistering on despawn). The [`App`](crate::app::App)
//! main loop calls it after every event/update batch.
//!
//! ## Multi-instance objects
//!
//! A multi-instance logical entity (e.g. a 7-segment digit, a text label with
//! many glyphs) keeps `Object::renderable = None` and stores its
//! [`ModelHandle`]s inside the behaviour. The behaviour contributes their
//! transforms each frame via [`Behaviour::collect_renderables`], and is
//! responsible for unregistering them in [`Behaviour::on_despawn`].
//!
//! ## Coordinate convention
//!
//! Positive Y is downwards in 2D, matching Vulkan clip space. Behaviours that
//! deal with directions ("paddle up") follow the same convention.

use std::any::Any;
use std::collections::HashMap;

use cgmath::{Matrix4, One, Quaternion, Vector3, Zero};
use winit::event::VirtualKeyCode;

use crate::graphics_manager::structures::hidden_transform;
use crate::graphics_manager::{GraphicsManager, MeshHandle, ModelHandle, TextureHandle};
use crate::resources::Resources;

/// Opaque identifier for an object in the [`Scene`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ObjectId(u32);

/// Position / rotation / scale, composed in that order into a model matrix.
/// Rotation is a quaternion so the same `Transform` can carry a 3D rotation
/// later; identity is the right default for 2D scenes.
#[derive(Clone, Copy)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Quaternion<f32>,
    pub scale: Vector3<f32>,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vector3::zero(),
            rotation: Quaternion::one(),
            scale: Vector3 { x: 1.0, y: 1.0, z: 1.0 },
        }
    }
}

impl Transform {
    pub fn from_position(position: Vector3<f32>) -> Self {
        Self { position, ..Self::default() }
    }

    pub fn matrix(&self) -> Matrix4<f32> {
        Matrix4::from_translation(self.position)
            * Matrix4::from(self.rotation)
            * Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z)
    }
}

/// What an `Object` draws, if anything. `None` is for logic-only objects
/// (controllers) and for multi-instance entities whose behaviour owns the
/// draws directly.
#[derive(Clone, Copy)]
pub enum Renderable {
    Solid { mesh: MeshHandle, color: [f32; 3] },
    Textured {
        texture: TextureHandle,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    },
}

/// Engine-side input event delivered to behaviours via [`Behaviour::on_event`].
/// Phase 3 will replace this with a proper polling `Input` plus a richer event
/// surface; for Phase 2 it just relays keyboard state changes.
#[derive(Clone, Copy)]
pub enum Event {
    KeyPressed(VirtualKeyCode),
    KeyReleased(VirtualKeyCode),
}

pub use winit::event::VirtualKeyCode as KeyCode;

/// Per-call context passed to behaviour hooks.
///
/// While a behaviour's hook is running, its own `behaviours` vec is empty
/// (the behaviour is "taken out" — see module docs), so `ctx.scene` exposes
/// every other object, plus the current object's transform/visible/renderable.
pub struct UpdateCtx<'a> {
    pub self_id: ObjectId,
    /// Frame delta for `update`. Zero for `on_event` calls.
    pub time: f32,
    pub scene: &'a mut Scene,
    pub resources: &'a mut Resources,
    pub graphics: &'a mut GraphicsManager,
    /// Behaviours set this to request graceful exit. The engine honours it
    /// at the next iteration of the event loop.
    pub exit_requested: &'a mut bool,
}

impl<'a> UpdateCtx<'a> {
    pub fn request_exit(&mut self) {
        *self.exit_requested = true;
    }
}

/// Game-logic unit attached to an `Object`. All hooks are optional; defaults
/// do nothing. `as_any_mut` / `as_any` enable typed cross-behaviour lookup
/// via [`Scene::behaviour_mut`] (e.g. a controller mutating a ball's velocity).
pub trait Behaviour: Any {
    fn update(&mut self, ctx: &mut UpdateCtx) {
        let _ = ctx;
    }
    fn on_event(&mut self, ctx: &mut UpdateCtx, event: &Event) {
        let _ = (ctx, event);
    }
    /// Contribute extra `(handle, transform)` pairs each frame. Used by
    /// multi-instance entities. `parent_matrix` is the owning object's
    /// `transform.matrix()` resolved already; behaviours typically post-multiply
    /// per-sub-instance local offsets onto it.
    fn collect_renderables(
        &self,
        parent_matrix: Matrix4<f32>,
        out: &mut Vec<(ModelHandle, Matrix4<f32>)>,
    ) {
        let _ = (parent_matrix, out);
    }
    /// Called when the parent object is despawned. Behaviours that registered
    /// extra `ModelHandle`s in their constructor must `unregister_*` them here.
    /// The `Object::renderable` instance is unregistered by the engine.
    fn on_despawn(&mut self, gm: &mut GraphicsManager) {
        let _ = gm;
    }

    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn as_any(&self) -> &dyn Any;
}

/// One node in the scene. Build via [`Object::new`] + the `with_*` chain;
/// hand to [`Scene::spawn`] which assigns its `ObjectId` and registers the
/// renderable instance at the next [`Scene::apply_commands`].
pub struct Object {
    pub transform: Transform,
    pub visible: bool,
    pub renderable: Option<Renderable>,
    pub behaviours: Vec<Box<dyn Behaviour>>,
    /// Set by the engine on insertion; the handle of the instance registered
    /// for `renderable`. `None` until apply_commands runs.
    model_handle: Option<ModelHandle>,
}

impl Default for Object {
    fn default() -> Self {
        Self::new()
    }
}

impl Object {
    pub fn new() -> Self {
        Self {
            transform: Transform::default(),
            visible: true,
            renderable: None,
            behaviours: Vec::new(),
            model_handle: None,
        }
    }

    pub fn with_transform(mut self, t: Transform) -> Self {
        self.transform = t;
        self
    }

    pub fn with_position(mut self, p: Vector3<f32>) -> Self {
        self.transform.position = p;
        self
    }

    pub fn with_renderable(mut self, r: Renderable) -> Self {
        self.renderable = Some(r);
        self
    }

    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    pub fn with_behaviour<B: Behaviour>(mut self, b: B) -> Self {
        self.behaviours.push(Box::new(b));
        self
    }

    pub fn model_handle(&self) -> Option<ModelHandle> {
        self.model_handle
    }
}

enum SceneCommand {
    Spawn(ObjectId, Object),
    Despawn(ObjectId),
}

pub struct Scene {
    objects: HashMap<ObjectId, Object>,
    next_id: u32,
    commands: Vec<SceneCommand>,
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            next_id: 0,
            commands: Vec::new(),
        }
    }

    /// Queue an object for insertion. The returned `ObjectId` is valid
    /// immediately for cross-wiring at build time, but `get(id)` won't
    /// resolve until the next [`apply_commands`](Self::apply_commands).
    pub fn spawn(&mut self, object: Object) -> ObjectId {
        let id = ObjectId(self.next_id);
        self.next_id += 1;
        self.commands.push(SceneCommand::Spawn(id, object));
        id
    }

    /// Queue an object for removal. The object's renderable is unregistered
    /// and every behaviour's `on_despawn` runs on the next apply_commands.
    pub fn despawn(&mut self, id: ObjectId) {
        self.commands.push(SceneCommand::Despawn(id));
    }

    pub fn get(&self, id: ObjectId) -> Option<&Object> {
        self.objects.get(&id)
    }

    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut Object> {
        self.objects.get_mut(&id)
    }

    pub fn contains(&self, id: ObjectId) -> bool {
        self.objects.contains_key(&id)
    }

    /// Find the first behaviour of type `T` on the given object. Returns
    /// `None` if the object doesn't exist or carries no `T`-typed behaviour.
    pub fn behaviour<T: Behaviour>(&self, id: ObjectId) -> Option<&T> {
        let obj = self.objects.get(&id)?;
        obj.behaviours
            .iter()
            .find_map(|b| b.as_any().downcast_ref::<T>())
    }

    pub fn behaviour_mut<T: Behaviour>(&mut self, id: ObjectId) -> Option<&mut T> {
        let obj = self.objects.get_mut(&id)?;
        obj.behaviours
            .iter_mut()
            .find_map(|b| b.as_any_mut().downcast_mut::<T>())
    }

    /// Drain pending spawn/despawn commands. Spawns register the object's
    /// `renderable` (if any) against the renderer and store the resulting
    /// `ModelHandle`. Despawns unregister it and run each behaviour's
    /// `on_despawn` hook.
    pub fn apply_commands(&mut self, gm: &mut GraphicsManager) {
        let cmds = std::mem::take(&mut self.commands);
        for cmd in cmds {
            match cmd {
                SceneCommand::Spawn(id, mut obj) => {
                    if let Some(r) = obj.renderable {
                        obj.model_handle = Some(match r {
                            Renderable::Solid { mesh, color } => {
                                gm.register_instance(mesh, color)
                            }
                            Renderable::Textured {
                                texture,
                                uv_offset,
                                uv_scale,
                            } => gm.register_textured_instance(texture, uv_offset, uv_scale),
                        });
                    }
                    self.objects.insert(id, obj);
                }
                SceneCommand::Despawn(id) => {
                    if let Some(mut obj) = self.objects.remove(&id) {
                        if let (Some(handle), Some(renderable)) =
                            (obj.model_handle, obj.renderable)
                        {
                            match renderable {
                                Renderable::Solid { .. } => gm.unregister_instance(handle),
                                Renderable::Textured { .. } => {
                                    gm.unregister_textured_instance(handle)
                                }
                            }
                        }
                        for b in obj.behaviours.iter_mut() {
                            b.on_despawn(gm);
                        }
                    }
                }
            }
        }
    }

    /// Build the `(ModelHandle, Matrix4)` list the renderer expects each frame.
    /// Walks every object, emits the renderable's instance (parked if
    /// `!visible`), and lets each behaviour contribute extra entries.
    pub fn collect_transforms(&self) -> Vec<(ModelHandle, Matrix4<f32>)> {
        let mut out = Vec::new();
        for obj in self.objects.values() {
            let m = obj.transform.matrix();
            if let Some(handle) = obj.model_handle {
                let drawn = if obj.visible { m } else { hidden_transform() };
                out.push((handle, drawn));
            }
            for b in obj.behaviours.iter() {
                b.collect_renderables(m, &mut out);
            }
        }
        out
    }

    /// Dispatch `update` to every behaviour. Order across objects is not
    /// guaranteed (HashMap iteration); within an object, behaviours run in
    /// their stored Vec order.
    pub(crate) fn dispatch_update(
        &mut self,
        dt: f32,
        resources: &mut Resources,
        gm: &mut GraphicsManager,
        exit_requested: &mut bool,
    ) {
        // Snapshot IDs so that mid-update spawns (which only land at
        // apply_commands anyway) don't affect this frame's dispatch set.
        let ids: Vec<ObjectId> = self.objects.keys().copied().collect();
        for id in ids {
            let mut taken = match self.objects.get_mut(&id) {
                Some(o) => std::mem::take(&mut o.behaviours),
                None => continue,
            };
            for b in taken.iter_mut() {
                let mut ctx = UpdateCtx {
                    self_id: id,
                    time: dt,
                    scene: self,
                    resources,
                    graphics: gm,
                    exit_requested,
                };
                b.update(&mut ctx);
            }
            if let Some(o) = self.objects.get_mut(&id) {
                o.behaviours = taken;
            }
        }
    }

    pub(crate) fn dispatch_event(
        &mut self,
        event: &Event,
        resources: &mut Resources,
        gm: &mut GraphicsManager,
        exit_requested: &mut bool,
    ) {
        let ids: Vec<ObjectId> = self.objects.keys().copied().collect();
        for id in ids {
            let mut taken = match self.objects.get_mut(&id) {
                Some(o) => std::mem::take(&mut o.behaviours),
                None => continue,
            };
            for b in taken.iter_mut() {
                let mut ctx = UpdateCtx {
                    self_id: id,
                    time: 0.0,
                    scene: self,
                    resources,
                    graphics: gm,
                    exit_requested,
                };
                b.on_event(&mut ctx, event);
            }
            if let Some(o) = self.objects.get_mut(&id) {
                o.behaviours = taken;
            }
        }
    }
}
