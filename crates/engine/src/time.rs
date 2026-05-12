//! Frame timing + fixed-timestep accumulator.
//!
//! [`Time`] is exposed to behaviours via `UpdateCtx::time`. It carries both a
//! variable per-frame delta (for rendering / animations in
//! [`Behaviour::update`](crate::scene::Behaviour::update)) and a constant fixed
//! step (for deterministic physics in
//! [`Behaviour::fixed_update`](crate::scene::Behaviour::fixed_update)).
//!
//! ## Loop shape
//!
//! Each frame, [`App`](crate::app::App) calls [`Time::begin_frame`] to advance
//! wall-clock state, then drains the accumulator with [`Time::consume_fixed_step`]
//! firing one `fixed_update` per step, then runs a single `update`. While
//! `fixed_update` is dispatching, [`Time::delta_time`] returns the fixed step;
//! during `update`, it returns the variable per-frame delta. `fixed_delta` is
//! always the constant (1/120s) regardless of phase.
//!
//! ## Spiral-of-death guard
//!
//! Frame deltas are clamped to [`MAX_FRAME_DT`] before being added to the
//! accumulator. A pause (debugger, OS hitch) won't cause an unbounded burst of
//! fixed steps on the next frame — at most ~30 catch-up steps at 120Hz.

use std::time::Instant;

/// Fixed-update rate in Hz. 120 was picked over 60 so the fixed step (~8.3 ms)
/// is comfortably smaller than the smallest collision feature (paddle width,
/// ~0.2 world units at typical speeds) — keeps the ball from tunnelling
/// through paddles without needing substep loops in physics behaviours.
pub const FIXED_HZ: f32 = 120.0;

/// Largest delta the loop will simulate in a single frame. If the host
/// process is paused longer than this (debugger, swap-out), the missed time
/// is dropped on the floor rather than fed to the accumulator.
pub const MAX_FRAME_DT: f32 = 0.25;

const FPS_SAMPLES: usize = 5;

pub struct Time {
    last_tick: Instant,
    frame_dt: f32,
    elapsed: f32,
    fixed_dt: f32,
    accumulator: f32,
    /// Delta visible to the current behaviour dispatch — set by
    /// [`set_phase_fixed`](Self::set_phase_fixed) /
    /// [`set_phase_variable`](Self::set_phase_variable).
    current_dt: f32,
    fps_samples: [u32; FPS_SAMPLES],
    fps_sample_idx: usize,
}

impl Default for Time {
    fn default() -> Self {
        Self::new()
    }
}

impl Time {
    pub fn new() -> Self {
        Self {
            last_tick: Instant::now(),
            frame_dt: 0.0,
            elapsed: 0.0,
            fixed_dt: 1.0 / FIXED_HZ,
            accumulator: 0.0,
            current_dt: 0.0,
            fps_samples: [0; FPS_SAMPLES],
            fps_sample_idx: 0,
        }
    }

    /// Delta seconds for the *current* dispatch phase.
    /// - In `update`: the variable per-frame delta (clamped by [`MAX_FRAME_DT`]).
    /// - In `fixed_update`: the constant [`fixed_delta`](Self::fixed_delta).
    pub fn delta_time(&self) -> f32 {
        self.current_dt
    }

    /// Constant fixed-step duration in seconds. Useful for planning ahead
    /// (e.g. computing how far an object will move next physics tick).
    pub fn fixed_delta(&self) -> f32 {
        self.fixed_dt
    }

    /// Wall-clock seconds since this `Time` was constructed (sum of clamped
    /// frame deltas — `MAX_FRAME_DT` skips count here too).
    pub fn elapsed(&self) -> f32 {
        self.elapsed
    }

    /// Smoothed frames-per-second over the last [`FPS_SAMPLES`] frames.
    pub fn fps(&self) -> f32 {
        let sum: u32 = self.fps_samples.iter().sum();
        if sum == 0 {
            return 0.0;
        }
        1_000_000.0 * FPS_SAMPLES as f32 / sum as f32
    }

    /// Begin a new frame: snapshot wall-clock delta, fold into the fixed-step
    /// accumulator, refresh FPS sample. Returns the (clamped) variable delta.
    pub(crate) fn begin_frame(&mut self) -> f32 {
        let now = Instant::now();
        let dur = now.duration_since(self.last_tick);
        self.last_tick = now;
        let micros = dur.as_secs() as u32 * 1_000_000 + dur.subsec_micros();
        self.fps_samples[self.fps_sample_idx] = micros;
        self.fps_sample_idx = (self.fps_sample_idx + 1) % FPS_SAMPLES;
        let dt = (micros as f32 / 1_000_000.0).min(MAX_FRAME_DT);
        self.frame_dt = dt;
        self.elapsed += dt;
        self.accumulator += dt;
        dt
    }

    /// Drain one fixed step from the accumulator if one is available. Caller
    /// loops on this between begin_frame and the variable update.
    pub(crate) fn consume_fixed_step(&mut self) -> bool {
        if self.accumulator >= self.fixed_dt {
            self.accumulator -= self.fixed_dt;
            true
        } else {
            false
        }
    }

    pub(crate) fn set_phase_fixed(&mut self) {
        self.current_dt = self.fixed_dt;
    }

    pub(crate) fn set_phase_variable(&mut self) {
        self.current_dt = self.frame_dt;
    }
}
