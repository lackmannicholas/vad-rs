pub mod energy;
pub mod onnx;
pub mod ten_vad;

pub trait VadModel: Send {
    /// Run inference on a single audio frame (expected to be f32 samples at the model's sample rate).
    fn predict(&mut self, frame: &[f32]) -> f32;
    fn reset(&mut self);
    fn sample_rate(&self) -> u32;
    fn frame_size(&self) -> usize;
}

pub fn instantiate_model(
    model_name: &str,
    input_sample_rate: u32,
) -> Result<Box<dyn VadModel + Send>, String> {
    match model_name {
        "energy" => Ok(Box::new(energy::EnergyModel::new())),
        "silero" => Ok(Box::new(onnx::SileroModel::new(input_sample_rate)?)),
        "ten_vad" => Ok(Box::new(ten_vad::TenVadModel::new()?)),
        _ => Err(format!("Unknown model: {}", model_name)),
    }
}
