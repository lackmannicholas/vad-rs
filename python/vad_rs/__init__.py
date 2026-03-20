import importlib.metadata
from typing import Iterator, Iterable, Optional, Tuple

from ._core import AudioFormat, VadResult, SpeechSegment, VoiceActivityDetectorCore as _CoreDetector

__version__ = importlib.metadata.version("vad-rs")

__all__ = [
    "VoiceActivityDetector",
    "AudioFormat",
    "VadResult",
    "SpeechSegment",
]


def _resolve_model(model: str) -> Tuple[str, Optional[str]]:
    if model == "energy":
        return "energy", None
    elif model == "silero":
        try:
            import vad_rs_silero
            return "silero", vad_rs_silero.model_path()
        except ImportError:
            raise ImportError(
                "The silero model is not installed. "
                "Install it with: pip install vad-rs[silero]"
            )
    elif model == "ten_vad":
        try:
            import vad_rs_ten_vad
            return "ten_vad", vad_rs_ten_vad.model_path()
        except ImportError:
            raise ImportError(
                "The ten_vad model is not installed. "
                "Install it with: pip install vad-rs[ten-vad]"
            )
    else:
        raise ValueError(
            f"Unknown model: {model!r}. "
            "Choose from: 'energy', 'silero', 'ten_vad'"
        )


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
        model_name, model_path = _resolve_model(model)
        self._core = _CoreDetector(model_name, model_path, input_sample_rate, input_format, threshold, min_speech_ms, min_silence_ms, hangover_ms)

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
