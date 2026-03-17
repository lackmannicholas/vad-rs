import pytest
import math
import struct
from vad_rs import VoiceActivityDetector, AudioFormat, SpeechSegment


def generate_sine_wave(freq: float, duration_ms: int, sample_rate: int, amplitude: float = 0.5) -> bytes:
    """Generate a 16-bit PCM sine wave for testing."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    samples = []
    for i in range(num_samples):
        # Sine wave formula
        t = i / sample_rate
        val = amplitude * math.sin(2 * math.pi * freq * t)
        # Convert [-1.0, 1.0] to 16-bit PCM scale
        pcm_val = int(val * 32767)
        samples.append(struct.pack('<h', pcm_val))
    return b''.join(samples)


def generate_silence(duration_ms: int, sample_rate: int) -> bytes:
    """Generate silent 16-bit PCM audio."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    return b'\x00\x00' * num_samples


def test_detector_initialization():
    vad = VoiceActivityDetector(model="energy", input_sample_rate=16000, input_format=AudioFormat.Pcm16, threshold=0.5)
    assert vad is not None


def test_process_frame_silence():
    vad = VoiceActivityDetector(model="energy", threshold=0.1)
    silence = generate_silence(100, 16000)  # 100ms of silence

    result = vad.process_frame(silence)
    assert not result.is_speech
    assert result.confidence < 0.1


def test_process_frame_speech():
    # Use energy model which triggers on RMS volume
    vad = VoiceActivityDetector(model="energy", threshold=0.1)
    loud_tone = generate_sine_wave(440.0, 100, 16000, amplitude=0.9)  # 100ms loud tone

    # Process multiple frames to get past min_speech_ms (250ms default)
    vad.process_frame(loud_tone)
    vad.process_frame(loud_tone)
    result = vad.process_frame(loud_tone)

    # Should trigger speech state with high confidence
    assert result.is_speech
    assert result.confidence >= 0.1


def test_process_stream_yields_segments():
    # Configure low timeouts for quick testing
    vad = VoiceActivityDetector(model="energy", threshold=0.1, min_speech_ms=50, min_silence_ms=50, hangover_ms=100)

    # Create a stream: 200ms silence -> 200ms tone -> 300ms silence
    silence1 = generate_silence(200, 16000)
    speech1 = generate_sine_wave(440.0, 200, 16000, amplitude=0.9)
    silence2 = generate_silence(300, 16000)

    # Break into 50ms chunks to simulate network stream
    stream = []
    for audio in [silence1, speech1, silence2]:
        chunk_size = int(16000 * (50 / 1000.0) * 2)  # frame size in bytes
        for i in range(0, len(audio), chunk_size):
            stream.append(audio[i : i + chunk_size])

    segments = list(vad.process_stream(stream))

    assert len(segments) == 1
    seg = segments[0]

    # 200ms silence at start.
    assert seg.start_ms >= 150 and seg.start_ms <= 250
    assert seg.end_ms >= 450 and seg.end_ms <= 550
    assert seg.confidence > 0.1


# ---------------------------------------------------------------------------
# Helpers for ML model tests (speech-like harmonics)
# ---------------------------------------------------------------------------


def generate_speech_harmonics(duration_ms: int, sample_rate: int, amplitude: float = 0.8) -> bytes:
    """Generate speech-like harmonics (fundamental + overtones) as 16-bit PCM.

    Uses amplitude modulation and harmonic roll-off to better approximate
    voiced speech, which models like Silero are trained to recognise.
    """
    import random

    random.seed(42)

    num_samples = int(sample_rate * (duration_ms / 1000.0))
    # Harmonic frequencies with natural roll-off (1/n weighting)
    harmonics = [(150, 1.0), (300, 0.5), (450, 0.33), (600, 0.25), (900, 0.15), (1200, 0.1), (1800, 0.06), (2400, 0.04)]

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Amplitude modulation at ~6 Hz (syllable rate) gives speech-like envelope
        env = 0.6 + 0.4 * math.sin(2 * math.pi * 6.0 * t)
        val = 0.0
        for f, w in harmonics:
            if f < sample_rate / 2:  # respect Nyquist
                val += w * math.sin(2 * math.pi * f * t)
        # Add slight noise (~5%) like glottal turbulence
        val += 0.05 * (random.random() * 2 - 1)
        val = val * env * amplitude
        pcm_val = max(-32768, min(32767, int(val * 32767)))
        samples.append(struct.pack('<h', pcm_val))
    return b''.join(samples)


# ---------------------------------------------------------------------------
# Silero VAD tests
# ---------------------------------------------------------------------------


class TestSilero:
    def test_initialization(self):
        vad = VoiceActivityDetector(model="silero", input_sample_rate=16000, input_format=AudioFormat.Pcm16)
        assert vad is not None

    def test_initialization_8khz(self):
        vad = VoiceActivityDetector(model="silero", input_sample_rate=8000, input_format=AudioFormat.Pcm16)
        assert vad is not None

    def test_silence_low_confidence(self):
        vad = VoiceActivityDetector(model="silero", input_sample_rate=16000, threshold=0.5)
        silence = generate_silence(500, 16000)

        # Process in chunks
        chunk_size = 512 * 2  # 512 samples = 32ms at 16kHz, ×2 for int16 bytes
        result = None
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            if len(chunk) == chunk_size:
                result = vad.process_frame(chunk)

        assert result is not None
        assert result.confidence < 0.3

    def test_speech_harmonics_detection(self):
        """Silero native 8kHz responds well to speech-like harmonics."""
        vad = VoiceActivityDetector(model="silero", input_sample_rate=8000, threshold=0.3)
        speech = generate_speech_harmonics(500, 8000)

        chunk_size = 256 * 2  # 256 samples = native 8kHz Silero frame
        max_conf = 0.0
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i : i + chunk_size]
            if len(chunk) == chunk_size:
                result = vad.process_frame(chunk)
                max_conf = max(max_conf, result.confidence)

        assert max_conf > 0.3, f"Expected speech detection > 0.3, got {max_conf}"

    def test_reset(self):
        vad = VoiceActivityDetector(model="silero", input_sample_rate=16000)
        speech = generate_speech_harmonics(200, 16000)

        chunk_size = 512 * 2
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i : i + chunk_size]
            if len(chunk) == chunk_size:
                vad.process_frame(chunk)

        vad.reset()

        # After reset, silence should produce low confidence
        silence = generate_silence(200, 16000)
        result = None
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            if len(chunk) == chunk_size:
                result = vad.process_frame(chunk)

        assert result is not None
        assert result.confidence < 0.3

    def test_process_stream(self):
        """Silero native 8kHz detects speech segments in a stream."""
        vad = VoiceActivityDetector(
            model="silero",
            input_sample_rate=8000,
            threshold=0.1,
            min_speech_ms=50,
            min_silence_ms=50,
            hangover_ms=50,
        )

        silence1 = generate_silence(500, 8000)
        speech = generate_speech_harmonics(1000, 8000)
        silence2 = generate_silence(1000, 8000)

        chunk_size = 256 * 2
        stream = []
        for audio in [silence1, speech, silence2]:
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if len(chunk) == chunk_size:
                    stream.append(chunk)

        segments = list(vad.process_stream(stream))
        # Should detect at least one speech segment
        assert len(segments) >= 1
        assert segments[0].confidence > 0.05


# ---------------------------------------------------------------------------
# TEN VAD tests
# ---------------------------------------------------------------------------


class TestTenVad:
    def test_initialization(self):
        vad = VoiceActivityDetector(model="ten_vad", input_sample_rate=16000, input_format=AudioFormat.Pcm16)
        assert vad is not None

    def test_initialization_8khz(self):
        """TEN VAD operates at 16kHz internally; pipeline should resample from 8kHz."""
        vad = VoiceActivityDetector(model="ten_vad", input_sample_rate=8000, input_format=AudioFormat.Pcm16)
        assert vad is not None

    def test_silence_low_confidence(self):
        vad = VoiceActivityDetector(model="ten_vad", input_sample_rate=16000, threshold=0.5)
        silence = generate_silence(500, 16000)

        chunk_size = 512 * 2
        result = None
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            if len(chunk) == chunk_size:
                result = vad.process_frame(chunk)

        assert result is not None
        assert result.confidence < 0.3

    def test_speech_harmonics_detection(self):
        vad = VoiceActivityDetector(model="ten_vad", input_sample_rate=16000, threshold=0.3)
        speech = generate_speech_harmonics(500, 16000)

        chunk_size = 512 * 2
        max_conf = 0.0
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i : i + chunk_size]
            if len(chunk) == chunk_size:
                result = vad.process_frame(chunk)
                max_conf = max(max_conf, result.confidence)

        assert max_conf > 0.3, f"Expected speech detection > 0.3, got {max_conf}"

    def test_reset(self):
        vad = VoiceActivityDetector(model="ten_vad", input_sample_rate=16000)
        speech = generate_speech_harmonics(200, 16000)

        chunk_size = 512 * 2
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i : i + chunk_size]
            if len(chunk) == chunk_size:
                vad.process_frame(chunk)

        vad.reset()

        silence = generate_silence(200, 16000)
        result = None
        for i in range(0, len(silence), chunk_size):
            chunk = silence[i : i + chunk_size]
            if len(chunk) == chunk_size:
                result = vad.process_frame(chunk)

        assert result is not None
        assert result.confidence < 0.3

    def test_process_stream(self):
        vad = VoiceActivityDetector(model="ten_vad", threshold=0.3, min_speech_ms=50, min_silence_ms=50, hangover_ms=100)

        silence1 = generate_silence(300, 16000)
        speech = generate_speech_harmonics(500, 16000)
        silence2 = generate_silence(500, 16000)

        chunk_size = 512 * 2
        stream = []
        for audio in [silence1, speech, silence2]:
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if len(chunk) == chunk_size:
                    stream.append(chunk)

        segments = list(vad.process_stream(stream))
        assert len(segments) >= 1
        assert segments[0].confidence > 0.2
