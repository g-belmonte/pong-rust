//! User-tunable game settings — winning score, ball speed, paddle speed.
//!
//! Lives in an `Rc<RefCell<Settings>>` threaded through every scene builder
//! that needs it (menu, settings screen, game). Persisted as JSON at
//! `dirs::config_dir()/pong-rust/settings.json`. Loaded once at startup
//! (`Settings::load`, falls back to defaults on miss or parse error) and
//! saved when the player leaves the Settings screen, returns to the menu
//! from a game, or picks Quit.
//!
//! Range and step sizes:
//! - winning_score: 1..=9 (discrete)
//! - ball_speed: 1.0..=8.0, step 0.5 (centred around the historical default 4.0)
//! - paddle_speed: 0.5..=4.0, step 0.5 (centred around 2.0)
//!
//! Changes take effect at the next `scene_game::build_game` call; a match
//! already in progress keeps its baked-in values.

use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

pub const WINNING_SCORE_MIN: u8 = 1;
pub const WINNING_SCORE_MAX: u8 = 9;
pub const BALL_SPEED_MIN: f32 = 1.0;
pub const BALL_SPEED_MAX: f32 = 8.0;
pub const PADDLE_SPEED_MIN: f32 = 0.5;
pub const PADDLE_SPEED_MAX: f32 = 4.0;
pub const SPEED_STEP: f32 = 0.5;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Settings {
    pub winning_score: u8,
    pub ball_speed: f32,
    pub paddle_speed: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            winning_score: 9,
            ball_speed: 4.0,
            paddle_speed: 2.0,
        }
    }
}

impl Settings {
    /// Disk location: `dirs::config_dir()/pong-rust/settings.json`. Returns
    /// `None` on platforms where `dirs::config_dir()` isn't defined
    /// (effectively all desktop platforms have it, but `dirs` is honest
    /// about it).
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("pong-rust").join("settings.json"))
    }

    /// Load from disk, falling back to `Default::default()` on any failure
    /// (no file, parse error, IO error). Always clamps fields into their
    /// valid ranges so a hand-edited file with bad values can't crash the
    /// game or break invariants. Failures are logged to stderr once.
    pub fn load() -> Self {
        let Some(path) = Self::config_path() else {
            return Self::default();
        };
        let Ok(bytes) = fs::read(&path) else {
            return Self::default();
        };
        match serde_json::from_slice::<Settings>(&bytes) {
            Ok(mut s) => {
                s.clamp();
                s
            }
            Err(e) => {
                eprintln!(
                    "settings: failed to parse {}: {} — using defaults",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
    }

    /// Best-effort write to disk. Creates the parent directory if missing.
    /// Logs failures to stderr but does not surface them — settings save is
    /// not on a critical path.
    pub fn save(&self) {
        let Some(path) = Self::config_path() else {
            return;
        };
        if let Some(parent) = path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!(
                    "settings: could not create {}: {}",
                    parent.display(),
                    e
                );
                return;
            }
        }
        let bytes = match serde_json::to_vec_pretty(self) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("settings: serialise failed: {}", e);
                return;
            }
        };
        if let Err(e) = fs::write(&path, bytes) {
            eprintln!("settings: write to {} failed: {}", path.display(), e);
        }
    }

    fn clamp(&mut self) {
        self.winning_score = self
            .winning_score
            .clamp(WINNING_SCORE_MIN, WINNING_SCORE_MAX);
        self.ball_speed = round_to_step(self.ball_speed)
            .clamp(BALL_SPEED_MIN, BALL_SPEED_MAX);
        self.paddle_speed = round_to_step(self.paddle_speed)
            .clamp(PADDLE_SPEED_MIN, PADDLE_SPEED_MAX);
    }

    pub fn inc_winning_score(&mut self) {
        self.winning_score = (self.winning_score + 1).min(WINNING_SCORE_MAX);
    }

    pub fn dec_winning_score(&mut self) {
        if self.winning_score > WINNING_SCORE_MIN {
            self.winning_score -= 1;
        }
    }

    pub fn inc_ball_speed(&mut self) {
        self.ball_speed = (self.ball_speed + SPEED_STEP).min(BALL_SPEED_MAX);
    }

    pub fn dec_ball_speed(&mut self) {
        self.ball_speed = (self.ball_speed - SPEED_STEP).max(BALL_SPEED_MIN);
    }

    pub fn inc_paddle_speed(&mut self) {
        self.paddle_speed = (self.paddle_speed + SPEED_STEP).min(PADDLE_SPEED_MAX);
    }

    pub fn dec_paddle_speed(&mut self) {
        self.paddle_speed = (self.paddle_speed - SPEED_STEP).max(PADDLE_SPEED_MIN);
    }
}

fn round_to_step(v: f32) -> f32 {
    (v / SPEED_STEP).round() * SPEED_STEP
}
