//! Audio: a thin wrapper over `kira`.
//!
//! [`AudioManager`] owns a `kira::AudioManager<DefaultBackend>` and is exposed
//! to behaviours through `UpdateCtx::audio`. Game code calls `audio.play(&sound)`;
//! the returned `StaticSoundHandle` (for volume/panning tweaks) is discarded —
//! callers that need it should reach for `kira` directly.
//!
//! [`Sound`] is an owning handle around a [`kira::sound::static_sound::StaticSoundData`].
//! It is loaded once via [`crate::resources::Resources::load_sound`] and cloned
//! by every consumer that wants to play it: `StaticSoundData` is internally
//! `Arc`-shared, so clones are cheap and do not duplicate the decoded samples.
//!
//! Unlike [`crate::resources::Mesh`] / [`crate::resources::Texture`], `Sound`
//! does not push anything into the pending-destroys queue: there is no GPU
//! resource to free under `device_wait_idle`, just heap memory that the inner
//! `Arc` cleans up when the last clone drops.
//!
//! [`AudioManager::new`] returns a handle even if the underlying device
//! initialisation fails (e.g. headless CI, no audio device); `play` is a no-op
//! in that case. Construction errors are logged once at startup.

use std::io::Cursor;

use kira::sound::static_sound::StaticSoundData;
use kira::sound::FromFileError;
use kira::{AudioManager as KiraAudioManager, AudioManagerSettings, DefaultBackend};

/// Decoded audio asset. Cheap to clone (internally `Arc`-shared by `kira`).
#[derive(Clone)]
pub struct Sound {
    data: StaticSoundData,
}

impl Sound {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, FromFileError> {
        // Copy bytes to an owned Vec so the Cursor is `'static + Send + Sync` —
        // `StaticSoundData::from_cursor` requires that bound.
        let owned: Vec<u8> = bytes.to_vec();
        let data = StaticSoundData::from_cursor(Cursor::new(owned))?;
        Ok(Self { data })
    }

    pub(crate) fn data(&self) -> StaticSoundData {
        self.data.clone()
    }
}

/// Engine-side audio output. Owned by `App` and threaded through `UpdateCtx`.
pub struct AudioManager {
    inner: Option<KiraAudioManager<DefaultBackend>>,
}

impl Default for AudioManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioManager {
    pub fn new() -> Self {
        let inner = match KiraAudioManager::<DefaultBackend>::new(AudioManagerSettings::default()) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("engine::audio: failed to initialise audio backend ({e:?}); playback disabled");
                None
            }
        };
        Self { inner }
    }

    /// Play `sound` once. Returns silently if the audio backend failed to
    /// initialise or if `kira` rejects the play call (queue full, etc.).
    pub fn play(&mut self, sound: &Sound) {
        if let Some(ref mut m) = self.inner {
            let _ = m.play(sound.data());
        }
    }
}
