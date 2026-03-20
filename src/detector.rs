use crate::inference::{instantiate_model, VadModel};
use crate::pipeline::AudioPipeline;
use crate::AudioFormat;

#[derive(Debug, Clone, Copy, PartialEq)]
enum VadState {
    Silence,
    PendingSpeech,
    Speech,
}

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start_ms: u32,
    pub end_ms: u32,
    pub confidence: f32,
}

pub struct VoiceActivityDetectorCore {
    model: Box<dyn VadModel + Send>,
    pipeline: AudioPipeline,
    input_format: AudioFormat,
    threshold: f32,

    // Config in model frames/counts
    min_speech_frames: u32,
    min_silence_frames: u32,
    hangover_frames: u32,
    frame_duration_ms: u32,

    // State machine
    state: VadState,
    state_frames_count: u32,
    last_confidence: f32,

    // Segment tracking
    total_frames_processed: u64,
    current_segment_start_frame: Option<u64>,
    current_segment_conf_sum: f32,
    current_segment_conf_count: u32,
}

impl VoiceActivityDetectorCore {
    pub fn new(
        model_name: &str,
        model_path: Option<&str>,
        input_sample_rate: u32,
        input_format: AudioFormat,
        threshold: f32,
        min_speech_ms: u32,
        min_silence_ms: u32,
        hangover_ms: u32,
    ) -> Result<Self, String> {
        let model = instantiate_model(model_name, model_path, input_sample_rate)?;
        let target_rate = model.sample_rate();
        let frame_size = model.frame_size();
        let pipeline = AudioPipeline::new(input_sample_rate, target_rate);

        // Calculate ms per frame for the model
        let frame_duration_ms = (frame_size as f32 / target_rate as f32 * 1000.0) as u32;
        let frame_duration_ms_nonzero = std::cmp::max(1, frame_duration_ms); // avoid div by zero

        let min_speech_frames = min_speech_ms / frame_duration_ms_nonzero;
        let min_silence_frames = min_silence_ms / frame_duration_ms_nonzero;
        let hangover_frames = hangover_ms / frame_duration_ms_nonzero;

        Ok(Self {
            model,
            pipeline,
            input_format,
            threshold,
            min_speech_frames,
            min_silence_frames,
            hangover_frames,
            frame_duration_ms: frame_duration_ms_nonzero,
            state: VadState::Silence,
            state_frames_count: 0,
            last_confidence: 0.0,
            total_frames_processed: 0,
            current_segment_start_frame: None,
            current_segment_conf_sum: 0.0,
            current_segment_conf_count: 0,
        })
    }

    pub fn process_frame(
        &mut self,
        audio_bytes: &[u8],
    ) -> Result<(bool, f32, Vec<SpeechSegment>), String> {
        let f32_samples = crate::audio::decode_audio(audio_bytes, &self.input_format)?;

        let frames = self.pipeline.process(f32_samples, self.model.frame_size());
        let mut segments = Vec::new();

        if frames.is_empty() {
            let is_speech = self.state == VadState::Speech;
            return Ok((is_speech, self.last_confidence, segments));
        }

        let mut final_confidence = self.last_confidence;

        for frame in frames {
            self.total_frames_processed += 1;
            let confidence = self.model.predict(&frame);
            final_confidence = confidence;

            if let Some(segment) = self.advance_state_machine(confidence) {
                segments.push(segment);
            }
        }

        self.last_confidence = final_confidence;

        let is_speech_now = self.state == VadState::Speech;
        Ok((is_speech_now, final_confidence, segments))
    }

    fn advance_state_machine(&mut self, confidence: f32) -> Option<SpeechSegment> {
        let is_above_threshold = confidence >= self.threshold;
        let mut completed_segment = None;

        match self.state {
            VadState::Silence => {
                if is_above_threshold {
                    self.state = VadState::PendingSpeech;
                    self.state_frames_count = 1;
                    self.current_segment_conf_sum = confidence;
                    self.current_segment_conf_count = 1;
                }
            }
            VadState::PendingSpeech => {
                if is_above_threshold {
                    self.state_frames_count += 1;
                    self.current_segment_conf_sum += confidence;
                    self.current_segment_conf_count += 1;
                    if self.state_frames_count >= self.min_speech_frames {
                        self.state = VadState::Speech;
                        // Segment includes the pending frames
                        self.current_segment_start_frame =
                            Some(self.total_frames_processed - self.state_frames_count as u64);
                        self.state_frames_count = 0;
                    }
                } else {
                    self.state = VadState::Silence;
                    self.state_frames_count = 0;
                    self.current_segment_conf_sum = 0.0;
                    self.current_segment_conf_count = 0;
                }
            }
            VadState::Speech => {
                self.current_segment_conf_sum += confidence;
                self.current_segment_conf_count += 1;

                if is_above_threshold {
                    self.state_frames_count = 0;
                } else {
                    self.state_frames_count += 1;
                    if self.state_frames_count >= self.min_silence_frames + self.hangover_frames {
                        if let Some(start_frame) = self.current_segment_start_frame {
                            let end_frame =
                                self.total_frames_processed - self.min_silence_frames as u64;
                            completed_segment = Some(SpeechSegment {
                                start_ms: (start_frame * self.frame_duration_ms as u64) as u32,
                                end_ms: (end_frame * self.frame_duration_ms as u64) as u32,
                                confidence: self.current_segment_conf_sum
                                    / self.current_segment_conf_count as f32,
                            });
                        }

                        self.state = VadState::Silence;
                        self.state_frames_count = 0;
                        self.current_segment_start_frame = None;
                        self.current_segment_conf_sum = 0.0;
                        self.current_segment_conf_count = 0;
                    }
                }
            }
        }

        completed_segment
    }

    pub fn reset(&mut self) {
        self.model.reset();
        self.pipeline.clear();
        self.state = VadState::Silence;
        self.state_frames_count = 0;
        self.last_confidence = 0.0;
        self.total_frames_processed = 0;
        self.current_segment_start_frame = None;
        self.current_segment_conf_sum = 0.0;
        self.current_segment_conf_count = 0;
    }
}
