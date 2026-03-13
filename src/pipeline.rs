use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub struct AudioPipeline {
    input_rate: u32,
    target_rate: u32,
    buffer: Vec<f32>,
    resampler: Option<SincFixedIn<f32>>,
}

impl AudioPipeline {
    pub fn new(input_rate: u32, target_rate: u32) -> Self {
        let resampler = if input_rate != target_rate {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            // 1024 is a reasonable default chunk size for input
            Some(
                SincFixedIn::<f32>::new(
                    target_rate as f64 / input_rate as f64,
                    2.0,
                    params,
                    1024,
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
            buffer: Vec::new(),
            resampler,
        }
    }

    /// Feeds new samples into the pipeline and returns a vector of fixed-size frames
    /// ready for the model to process.
    pub fn process(&mut self, mut samples: Vec<f32>, frame_size: usize) -> Vec<Vec<f32>> {
        // Resample if necessary
        if let Some(resampler) = &mut self.resampler {
            let mut resampled_samples = Vec::new();
            // Process chunks
            let chunk_size = resampler.input_frames_max();
            for chunk in samples.chunks(chunk_size) {
                let mut pad_chunk = chunk.to_vec();
                if pad_chunk.len() < chunk_size {
                    pad_chunk.resize(chunk_size, 0.0);
                }
                if let Ok(mut out) = resampler.process(&[pad_chunk], None) {
                    // Extract mono
                    resampled_samples.append(&mut out[0]);
                }
            }
            samples = resampled_samples;
        }

        self.buffer.extend(samples);

        let mut frames = Vec::new();
        while self.buffer.len() >= frame_size {
            let frame: Vec<f32> = self.buffer.drain(..frame_size).collect();
            frames.push(frame);
        }

        frames
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
        if let Some(resampler) = &mut self.resampler {
            resampler.reset();
        }
    }
}
