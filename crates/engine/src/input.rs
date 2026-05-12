//! Polling input state — keyboard + mouse.
//!
//! [`Input`] is exposed to behaviours via `UpdateCtx::input`. Game code reads
//! key/button state with [`is_pressed`](Input::is_pressed),
//! [`was_just_pressed`](Input::was_just_pressed),
//! [`was_just_released`](Input::was_just_released), plus the mouse equivalents.
//! No engine event surface for input exists post-Phase 3 — there is one input
//! path, and it is polling.
//!
//! ## Edge state lifetime
//!
//! `was_just_pressed` / `was_just_released` are set when the engine receives a
//! winit event between frames. They stay observable through *every*
//! `fixed_update` and the single `update` of the current frame, then are
//! cleared once at end-of-frame ([`end_frame`](Input::end_frame)). A single
//! Space tap is therefore visible to all of the frame's substeps, but never
//! survives into the next frame.
//!
//! ## Focus loss
//!
//! When the window loses focus, [`lose_focus`](Input::lose_focus) drops every
//! held key/button and synthesises release edges for them. Without this, an
//! Alt-Tab during a held key leaves the game thinking the key is still down
//! (winit doesn't deliver the release event to an unfocused window).

use std::collections::HashSet;

pub use winit::event::{MouseButton, VirtualKeyCode as KeyCode};

#[derive(Default)]
pub struct Input {
    pressed: HashSet<KeyCode>,
    just_pressed: HashSet<KeyCode>,
    just_released: HashSet<KeyCode>,

    mouse_x: f32,
    mouse_y: f32,
    mouse_pressed: HashSet<MouseButton>,
    mouse_just_pressed: HashSet<MouseButton>,
    mouse_just_released: HashSet<MouseButton>,
}

impl Input {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_pressed(&self, key: KeyCode) -> bool {
        self.pressed.contains(&key)
    }

    pub fn was_just_pressed(&self, key: KeyCode) -> bool {
        self.just_pressed.contains(&key)
    }

    pub fn was_just_released(&self, key: KeyCode) -> bool {
        self.just_released.contains(&key)
    }

    /// Cursor position in physical pixels relative to the window's top-left.
    /// Behaviour is `(0.0, 0.0)` until the first `CursorMoved` arrives.
    pub fn mouse_position(&self) -> (f32, f32) {
        (self.mouse_x, self.mouse_y)
    }

    pub fn is_mouse_pressed(&self, b: MouseButton) -> bool {
        self.mouse_pressed.contains(&b)
    }

    pub fn was_mouse_just_pressed(&self, b: MouseButton) -> bool {
        self.mouse_just_pressed.contains(&b)
    }

    pub fn was_mouse_just_released(&self, b: MouseButton) -> bool {
        self.mouse_just_released.contains(&b)
    }

    pub(crate) fn on_key_pressed(&mut self, k: KeyCode) {
        // winit emits repeat events while a key is held; only mark the
        // edge on the first transition into pressed.
        if self.pressed.insert(k) {
            self.just_pressed.insert(k);
        }
    }

    pub(crate) fn on_key_released(&mut self, k: KeyCode) {
        if self.pressed.remove(&k) {
            self.just_released.insert(k);
        }
    }

    pub(crate) fn on_mouse_moved(&mut self, x: f32, y: f32) {
        self.mouse_x = x;
        self.mouse_y = y;
    }

    pub(crate) fn on_mouse_pressed(&mut self, b: MouseButton) {
        if self.mouse_pressed.insert(b) {
            self.mouse_just_pressed.insert(b);
        }
    }

    pub(crate) fn on_mouse_released(&mut self, b: MouseButton) {
        if self.mouse_pressed.remove(&b) {
            self.mouse_just_released.insert(b);
        }
    }

    /// Clear edge-triggered state. Called once per frame, after the variable
    /// update — so every fixed_update and the update see the same edges.
    pub(crate) fn end_frame(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
        self.mouse_just_pressed.clear();
        self.mouse_just_released.clear();
    }

    /// Synthesise release edges for everything currently held, then drop the
    /// held state. Called on focus-loss so paddles don't get stuck moving.
    pub(crate) fn lose_focus(&mut self) {
        for k in self.pressed.drain() {
            self.just_released.insert(k);
        }
        for b in self.mouse_pressed.drain() {
            self.mouse_just_released.insert(b);
        }
    }
}
