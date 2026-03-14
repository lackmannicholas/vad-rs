use super::VadModel;

pub struct EnergyModel {
    sample_rate: u32,
    frame_size: usize,
}

impl EnergyModel {
    pub fn new() -> Self {
        Self {
            sample_rate: 16000,
            frame_size: 512, // example frame size
        }
    }
}

impl VadModel for EnergyModel {
    fn predict(&mut self, frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return 0.0;
        }

        // RMS energy calculation
        let mut sum_squares = 0.0;
        for &sample in frame {
            sum_squares += sample * sample;
        }
        let rms = (sum_squares / frame.len() as f32).sqrt();

        // Map RMS to a 0.0–1.0 confidence score using a logarithmic scale.
        // Telephony speech (µ-law, handset mic) typically has RMS of 0.02–0.15.
        // A loud speaker might reach 0.2–0.3. Background noise sits around 0.005–0.02.
        //
        // We use a dB-inspired mapping:
        //   - floor at -60 dB (RMS ~0.001) → 0.0
        //   - ceiling at -6 dB (RMS ~0.5)  → 1.0
        // This gives good separation across the full dynamic range.
        let floor_db: f32 = -60.0;
        let ceil_db: f32 = -6.0;
        let rms_db = if rms > 1e-10 {
            20.0 * rms.log10()
        } else {
            floor_db
        };
        ((rms_db - floor_db) / (ceil_db - floor_db)).clamp(0.0, 1.0)
    }

    fn reset(&mut self) {
        // No persistent state to reset in a simple RMS model
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn frame_size(&self) -> usize {
        self.frame_size
    }
}
