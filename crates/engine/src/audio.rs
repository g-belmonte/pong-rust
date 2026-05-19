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

use std::cell::RefCell;
use std::io::Cursor;
use std::rc::Rc;

use kira::sound::static_sound::StaticSoundData;
use kira::sound::FromFileError;
use kira::{AudioManager as KiraAudioManager, AudioManagerSettings, DefaultBackend};

/// Decoded audio asset. Cheap to clone — clones share an inner [`Rc`] so a
/// hot-reload of the source file (when the engine's `hot-reload` feature is
/// enabled) swaps the decoded data inside, and every outstanding clone picks
/// up the new sample on its next `play`.
#[derive(Clone)]
pub struct Sound {
    inner: Rc<RefCell<StaticSoundData>>,
}

impl Sound {
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, FromFileError> {
        let data = decode(bytes)?;
        Ok(Self {
            inner: Rc::new(RefCell::new(data)),
        })
    }

    pub(crate) fn data(&self) -> StaticSoundData {
        // kira's StaticSoundData is internally Arc-shared; `clone` is cheap
        // and does not duplicate the decoded samples.
        self.inner.borrow().clone()
    }

    /// Re-decode `bytes` and replace this `Sound`'s inner data in place. All
    /// outstanding clones of this `Sound` see the new sample on their next
    /// `play` call. Used by the asset watcher; no-op for game code.
    #[cfg(feature = "hot-reload")]
    pub(crate) fn replace_from_bytes(&self, bytes: &[u8]) -> Result<(), FromFileError> {
        let new_data = decode(bytes)?;
        *self.inner.borrow_mut() = new_data;
        Ok(())
    }

    /// Pointer-identity for hot-reload bookkeeping. Two `Sound`s share the
    /// same inner iff their `id()` matches.
    #[cfg(feature = "hot-reload")]
    pub(crate) fn id(&self) -> *const RefCell<StaticSoundData> {
        Rc::as_ptr(&self.inner)
    }

    /// Strong refcount of the inner cell. The watcher uses this to drop
    /// stale entries whose only remaining ref is the one it itself holds —
    /// i.e. game code dropped the sound. `> 1` means at least one game-side
    /// clone is still around.
    #[cfg(feature = "hot-reload")]
    pub(crate) fn strong_count(&self) -> usize {
        Rc::strong_count(&self.inner)
    }
}

fn decode(bytes: &[u8]) -> Result<StaticSoundData, FromFileError> {
    // Copy bytes to an owned Vec so the Cursor is `'static + Send + Sync` —
    // `StaticSoundData::from_cursor` requires that bound.
    let owned: Vec<u8> = bytes.to_vec();
    StaticSoundData::from_cursor(Cursor::new(owned))
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
