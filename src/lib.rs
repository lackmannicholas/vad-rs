use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod audio;
mod detector;
mod inference;
mod pipeline;

use detector::SpeechSegment as CoreSpeechSegment;
use detector::VoiceActivityDetectorCore;

#[derive(Clone, PartialEq, Eq)]
#[pyclass(name = "AudioFormat", eq, eq_int)]
pub enum AudioFormat {
    Pcm16 = 0,
    PcmF32 = 1,
    Ulaw = 2,
    Alaw = 3,
}

#[pyclass(name = "VadResult")]
pub struct VadResult {
    #[pyo3(get)]
    pub is_speech: bool,
    #[pyo3(get)]
    pub confidence: f32,
}

#[pyclass(name = "SpeechSegment")]
#[derive(Clone)]
pub struct SpeechSegment {
    #[pyo3(get)]
    pub start_ms: u32,
    #[pyo3(get)]
    pub end_ms: u32,
    #[pyo3(get)]
    pub confidence: f32,
}

#[pyclass(name = "VoiceActivityDetectorCore")]
pub struct VoiceActivityDetectorCorePy {
    inner: VoiceActivityDetectorCore,
}

#[pymethods]
impl VoiceActivityDetectorCorePy {
    #[new]
    #[pyo3(signature = (model="silero", input_sample_rate=16000, input_format=AudioFormat::Pcm16, threshold=0.5, min_speech_ms=250, min_silence_ms=100, hangover_ms=300))]
    fn new(
        model: &str,
        input_sample_rate: u32,
        input_format: AudioFormat,
        threshold: f32,
        min_speech_ms: u32,
        min_silence_ms: u32,
        hangover_ms: u32,
    ) -> PyResult<Self> {
        let inner = VoiceActivityDetectorCore::new(
            model,
            input_sample_rate,
            input_format,
            threshold,
            min_speech_ms,
            min_silence_ms,
            hangover_ms,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

        Ok(Self { inner })
    }

    fn process_frame(
        &mut self,
        audio_bytes: &Bound<'_, PyBytes>,
    ) -> PyResult<(VadResult, Vec<SpeechSegment>)> {
        let bytes = audio_bytes.as_bytes();
        let (is_speech, confidence, raw_segments) = self
            .inner
            .process_frame(bytes)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let segments = raw_segments
            .into_iter()
            .map(|s| SpeechSegment {
                start_ms: s.start_ms,
                end_ms: s.end_ms,
                confidence: s.confidence,
            })
            .collect();

        Ok((
            VadResult {
                is_speech,
                confidence,
            },
            segments,
        ))
    }

    fn reset(&mut self) -> PyResult<()> {
        self.inner.reset();
        Ok(())
    }
}

/// A production-grade Voice Activity Detection (VAD) library.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<AudioFormat>()?;
    m.add_class::<VoiceActivityDetectorCorePy>()?;
    m.add_class::<VadResult>()?;
    m.add_class::<SpeechSegment>()?;
    Ok(())
}
