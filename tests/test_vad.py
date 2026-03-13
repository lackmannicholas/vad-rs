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
