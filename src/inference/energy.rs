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
        
        // Simple RMS energy calculation
        let mut sum_squares = 0.0;
        for &sample in frame {
            sum_squares += sample * sample;
        }
        let rms = (sum_squares / frame.len() as f32).sqrt();
        
        // Return a normalized value roughly mimicking a confidence score
        // (Just a placeholder for testing without ONNX)
        let max_energy = 0.5; // Arbitrary max energy
        (rms / max_energy).clamp(0.0, 1.0)
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