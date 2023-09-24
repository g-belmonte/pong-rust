use std::time::Instant;

const SAMPLE_COUNT: usize = 5;
const SAMPLE_COUNT_FLOAT: f32 = SAMPLE_COUNT as f32;

pub struct FPSLimiter {
    counter: Instant,
    samples: [u32; SAMPLE_COUNT],
    current_frame: usize,
    delta_frame: u32,
}

impl FPSLimiter {
    pub fn new() -> FPSLimiter {
        FPSLimiter {
            counter: Instant::now(),
            samples: [0; SAMPLE_COUNT],
            current_frame: 0,
            delta_frame: 0,
        }
    }

    /// Call this function in game loop to update its inner status.
    pub fn tick_frame(&mut self) {
        let time_elapsed = self.counter.elapsed();
        self.counter = Instant::now();

        self.delta_frame = time_elapsed.subsec_micros();
        self.samples[self.current_frame] = self.delta_frame;
        self.current_frame = (self.current_frame + 1) % SAMPLE_COUNT;
    }

    /// Calculate the current FPS.
    pub fn fps(&self) -> f32 {
        let mut sum = 0_u32;
        self.samples.iter().for_each(|val| {
            sum += val;
        });

        1_000_000.0_f32 / (sum as f32 / SAMPLE_COUNT_FLOAT)
    }

    /// Return current delta time in seconds
    /// this function ignore its second part, since the second is mostly zero.
    pub fn delta_time(&self) -> f32 {
        self.delta_frame as f32 / 1_000_000.0_f32 // time in second
    }
}
