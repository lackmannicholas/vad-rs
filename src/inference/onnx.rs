use super::VadModel;
use ort::session::Session;
use ort::value::Tensor;

pub struct SileroModel {
    session: Session,
    sample_rate: u32,
    frame_size: usize,
    /// RNN hidden state: shape [2, 1, 128]
    state: Vec<f32>,
}

impl SileroModel {
    pub fn new(model_path: &str, input_sample_rate: u32) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("Failed to create ORT session builder: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load Silero ONNX model from '{}': {}", model_path, e))?;

        // Silero VAD v5 supports both 8kHz and 16kHz natively.
        // At 8kHz: 256 samples per frame (32ms)
        // At 16kHz: 512 samples per frame (32ms)
        // Using native sample rate avoids resampling artifacts.
        let (sample_rate, frame_size) = match input_sample_rate {
            8000 => (8000, 256),
            _ => (16000, 512),
        };

        Ok(Self {
            session,
            sample_rate,
            frame_size,
            state: vec![0.0_f32; 2 * 1 * 128],
        })
    }
}

impl VadModel for SileroModel {
    fn predict(&mut self, frame: &[f32]) -> f32 {
        // Build input tensor: shape [1, frame_size]
        let input_data: Vec<f32> = frame.to_vec();
        let input_tensor = match Tensor::from_array((
            [1_i64, self.frame_size as i64],
            input_data.into_boxed_slice(),
        )) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };

        // Build state tensor: shape [2, 1, 128]
        let state_tensor = match Tensor::from_array((
            [2_i64, 1_i64, 128_i64],
            self.state.clone().into_boxed_slice(),
        )) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };

        // Build sample rate tensor: shape [1] i64 (Silero expects a 1-D tensor, not scalar)
        let sr_tensor = match Tensor::from_array(([1_i64], vec![self.sample_rate as i64])) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };

        // Run inference
        let outputs = match self.session.run(ort::inputs! {
            "input" => input_tensor,
            "state" => state_tensor,
            "sr" => sr_tensor,
        }) {
            Ok(o) => o,
            Err(_) => return 0.0,
        };

        // Extract speech probability from "output" (shape [1, 1])
        let confidence = match outputs["output"].try_extract_tensor::<f32>() {
            Ok((_shape, data)) => data[0],
            Err(_) => return 0.0,
        };

        // Update RNN state from "stateN" (shape [2, 1, 128])
        if let Ok((_shape, data)) = outputs["stateN"].try_extract_tensor::<f32>() {
            self.state = data.to_vec();
        }

        confidence
    }

    fn reset(&mut self) {
        self.state = vec![0.0_f32; 2 * 1 * 128];
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn frame_size(&self) -> usize {
        self.frame_size
    }
}
