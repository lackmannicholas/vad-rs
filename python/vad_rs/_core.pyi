from enum import IntEnum
from typing import Iterator, Iterable, Tuple, List

class AudioFormat(IntEnum):
    Pcm16 = 0
    PcmF32 = 1
    Ulaw = 2
    Alaw = 3

class VadResult:
    @property
    def is_speech(self) -> bool: ...
    @property
    def confidence(self) -> float: ...

class SpeechSegment:
    @property
    def start_ms(self) -> int: ...
    @property
    def end_ms(self) -> int: ...
    @property
    def confidence(self) -> float: ...

class VoiceActivityDetectorCore:
    def __init__(
        self,
        model: str = "silero",
        input_sample_rate: int = 16000,
        input_format: AudioFormat = AudioFormat.Pcm16,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        hangover_ms: int = 300,
    ) -> None: ...
    def process_frame(self, audio_bytes: bytes) -> Tuple[VadResult, List[SpeechSegment]]: ...
    def reset(self) -> None: ...

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
    ) -> None: ...
    def process_frame(self, audio_bytes: bytes) -> VadResult: ...
    def process_stream(self, frame_iterator: Iterable[bytes]) -> Iterator[SpeechSegment]: ...
    def reset(self) -> None: ...
