from typing import Iterator, Iterable

from ._core import AudioFormat, VadResult, SpeechSegment, VoiceActivityDetectorCore as _CoreDetector

__all__ = [
    "VoiceActivityDetector",
    "AudioFormat",
    "VadResult",
    "SpeechSegment",
]


class VoiceActivityDetector:
    def __init__(
        self,
        model: str = "silero",
        input_sample_rate: int = 16000,
        input_format: AudioFormat = AudioFormat.Pcm16,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        hangover_ms: int = 300,
    ) -> None:
        self._core = _CoreDetector(model, input_sample_rate, input_format, threshold, min_speech_ms, min_silence_ms, hangover_ms)

    def process_frame(self, audio_bytes: bytes) -> VadResult:
        result, _segments = self._core.process_frame(audio_bytes)
        return result

    def process_stream(self, frame_iterator: Iterable[bytes]) -> Iterator[SpeechSegment]:
        """
        Processes an iterable of audio bytes and yields bounded speech segments.
        """
        for frame in frame_iterator:
            _result, segments = self._core.process_frame(frame)
            for seg in segments:
                yield seg

    def reset(self) -> None:
        self._core.reset()
