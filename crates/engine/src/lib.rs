// `vk::*CreateInfo` structs carry a private `_marker: PhantomData` field that
// callers can't name; `..Default::default()` is the only way to fill it.
// Clippy doesn't see the private field and reports a false-positive
// `needless_update` on every such literal.
#![allow(clippy::needless_update)]

pub mod app;
pub mod audio;
pub mod camera;
pub mod graphics_manager;
pub mod input;
pub mod resources;
pub mod scene;
pub mod time;

pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
