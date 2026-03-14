use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub struct AudioPipeline {
    input_rate: u32,
    target_rate: u32,
    /// Pre-resampling buffer: accumulates decoded samples until we have
    /// enough for the resampler's fixed input chunk size.
    pre_resample_buffer: Vec<f32>,
    /// Post-resampling buffer: accumulates resampled samples until we have
    /// enough for a full model frame.
    buffer: Vec<f32>,
    resampler: Option<SincFixedIn<f32>>,
}

impl AudioPipeline {
    pub fn new(input_rate: u32, target_rate: u32) -> Self {
        let resampler = if input_rate != target_rate {
            let params = SincInterpolationParameters {
                sinc_len: 64,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 128,
                window: WindowFunction::BlackmanHarris2,
            };
            // Use 160 as chunk size to match typical telephony packet size
            // (20ms at 8kHz). This keeps resampling latency low (~20ms)
            // while still producing high-quality output for VAD purposes.
            Some(
                SincFixedIn::<f32>::new(
                    target_rate as f64 / input_rate as f64,
                    2.0,
                    params,
                    160,
                    1,
                )
                .expect("Failed to create resampler"),
            )
        } else {
            None
        };

        Self {
            input_rate,
            target_rate,
            pre_resample_buffer: Vec::new(),
            buffer: Vec::new(),
            resampler,
        }
    }

    /// Feeds new samples into the pipeline and returns a vector of fixed-size frames
    /// ready for the model to process.
    pub fn process(&mut self, samples: Vec<f32>, frame_size: usize) -> Vec<Vec<f32>> {
        // If no resampling needed, just buffer directly
        if self.resampler.is_none() {
            self.buffer.extend(samples);
        } else {
            // Buffer incoming samples pre-resampling
            self.pre_resample_buffer.extend(samples);

            let resampler = self.resampler.as_mut().unwrap();
            let chunk_size = resampler.input_frames_max();

            // Only feed the resampler when we have full chunks
            while self.pre_resample_buffer.len() >= chunk_size {
                let chunk: Vec<f32> = self.pre_resample_buffer.drain(..chunk_size).collect();
                if let Ok(mut out) = resampler.process(&[chunk], None) {
                    self.buffer.append(&mut out[0]);
                }
            }
        }

        let mut frames = Vec::new();
        while self.buffer.len() >= frame_size {
            let frame: Vec<f32> = self.buffer.drain(..frame_size).collect();
            frames.push(frame);
        }

        frames
    }

    pub fn clear(&mut self) {
        self.pre_resample_buffer.clear();
        self.buffer.clear();
        if let Some(resampler) = &mut self.resampler {
            resampler.reset();
        }
    }
}
