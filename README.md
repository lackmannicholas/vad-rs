# vad-rs

**vad-rs** is a production-grade Voice Activity Detection (VAD) library implemented in Rust with a clean, high-performance Python API. 

The Rust core handles audio preprocessing, frame management, audio resampling, and model inference, while Python consumers get an easy-to-use, native extension via PyO3 and Maturin.

## Features

- **Blazing Fast**: Core logic, state machines, and formatting run in Rust.
- **Multiple Models**: 
  - `silero`: (Default) Uses the ONNX runtime to run the highly accurate Silero VAD v5 model. Supports native 8kHz and 16kHz inference.
  - `ten_vad`: TEN Framework's ONNX VAD model with built-in STFT/mel-filterbank preprocessing. Operates at 16kHz with 16ms frame resolution.
  - `energy`: A lightweight RMS energy fallback (dB-scale) that requires no ML dependencies.
- **Versatile Format Support**: Built-in support for multiple raw audio formats:
  - `AudioFormat.Pcm16` (16-bit signed PCM)
  - `AudioFormat.PcmF32` (32-bit floating point PCM)
  - `AudioFormat.Ulaw` (8-bit G.711 μ-law) *Great for telephony!*
- **Any Sample Rate**: Pass in audio at 8kHz, 16kHz, 48kHz, etc. Resampling is handled internally.
- **Real-Time & Streaming**: Frame-by-frame processing or high-level segment iteration.

---

## Installation

Currently, this library is built from source using `maturin`.

### Prerequisites
- Python 3.9+
- Rust & Cargo installed ([rustup.rs](https://rustup.rs/))

### Build & Install

```bash
# Clone the repository
git clone https://github.com/lackmannicholas/vad-rs.git
cd vad-rs

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Build and install (using uv + maturin)
uv run maturin develop

# Or with pip + maturin directly:
# pip install maturin
# maturin develop --release
```

*(Note: Whenever you modify the underlying Rust code, you must re-run `maturin develop` to recompile the extension.)*

---

## Usage

### 1. Frame-by-Frame Processing (Real-Time)

Perfect for real-time applications like WebRTC or WebSockets where you process audio in small, continuous chunks as it arrives.

```python
from vad_rs import VoiceActivityDetector, AudioFormat

# Initialize the VAD detector
vad = VoiceActivityDetector(
    model="ten_vad",                     # "silero", "ten_vad", or "energy"
    input_sample_rate=8000,              # e.g., 8000 for telephony, 16000 for standard
    input_format=AudioFormat.Ulaw,       # AudioFormat.Pcm16, PcmF32, or Ulaw
    threshold=0.5,                       # Speech probability cutoff (0.0 to 1.0)
    min_speech_ms=250,                   # Min duration to count as speech (debounce)
    min_silence_ms=100,                  # Min silence before ending a segment
    hangover_ms=300,                     # Extra padding after speech ends
)

def handle_incoming_audio(audio_bytes: bytes):
    # Process a chunk of audio
    result = vad.process_frame(audio_bytes)
    
    if result.is_speech:
        print(f"🎤 Speech detected (Confidence: {result.confidence:.2f})")
    else:
        print(f"🔇 Silence (Confidence: {result.confidence:.2f})")

# If you need to clear the internal state (e.g., between different calls):
# vad.reset()
```

### 2. Streaming Segment Detection

Ideal for processing entire files or large continuous buffers when you just want to extract the boundaries of the spoken segments.

```python
from vad_rs import VoiceActivityDetector, AudioFormat

vad = VoiceActivityDetector(
    model="ten_vad",
    input_sample_rate=16000,
    input_format=AudioFormat.Pcm16
)

# frame_iterator can be any iterable that yields `bytes`
# e.g., a generator reading from a .wav file or a network socket
def read_frames(filepath):
    chunk_size = 512  # adjust based on your needs
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield chunk

for segment in vad.process_stream(read_frames("audio.raw")):
    print(
        f"Speech Segment: {segment.start_ms}ms to {segment.end_ms}ms "
        f"(Average Confidence: {segment.confidence:.2f})"
    )
```

---

## Models

### Silero VAD (`model="silero"`)

[Silero VAD v5](https://github.com/snakers4/silero-vad) is a widely-used, high-accuracy neural VAD. vad-rs embeds the ONNX model and runs it via `ort` (ONNX Runtime). Supports native 8kHz (256-sample frames) and 16kHz (512-sample frames) inference — no resampling needed when your input matches.

**Notes on synthetic/test signals**: Silero is trained on real speech and can be finicky with synthetic test signals (pure tones, simple harmonics). It may produce low confidence on generated audio that doesn't closely resemble human speech. This is expected — the model is optimized for real-world speech detection, not tone detection. In live use with real microphone audio, Silero performs very well.

### TEN VAD (`model="ten_vad"`)

[TEN VAD](https://github.com/TEN-framework/ten-vad) is from the TEN Framework. Unlike Silero, the ONNX model expects preprocessed features rather than raw audio, so vad-rs implements the full feature extraction pipeline in Rust:

1. **Pre-emphasis** (0.97 coefficient)
2. **STFT** (768-point Hann window, 1024-point FFT, 256-sample hop)
3. **Mel filterbank** (40 bands, 0–8 kHz)
4. **Pitch estimation** (autocorrelation, 60–500 Hz)
5. **Feature normalization** (hardcoded mean/std from training data)
6. **3-frame context stacking** → `[1, 3, 41]` input tensor

Operates at 16kHz only (pipeline resamples automatically from other rates). Frame size is 256 samples (16ms). In testing, TEN VAD responds more reliably to synthetic signals than Silero, which bodes well for less finicky behavior in production.

### Energy (`model="energy"`)

A zero-dependency RMS energy detector using dB-scale mapping. No neural network — just measures loudness. Floor at -60 dB, ceiling at -6 dB. Useful as a lightweight fallback or for testing pipeline integration without ML overhead.

---

## Configuration Parameters

The `VoiceActivityDetector` takes several tuning parameters in its constructor to adapt to your environment:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"silero"` | The backend model to use. `"silero"`, `"ten_vad"`, or `"energy"`. |
| `input_sample_rate` | `int` | `16000` | The sample rate of the input audio in Hz (e.g., `8000`, `16000`, `48000`). |
| `input_format` | `AudioFormat` | `AudioFormat.Pcm16` | The payload format (`AudioFormat.Pcm16`, `AudioFormat.PcmF32`, `AudioFormat.Ulaw`). |
| `threshold` | `float` | `0.5` | The probability threshold (0.0 - 1.0) above which a frame represents speech. |
| `min_speech_ms` | `int` | `250` | Minimum continuous duration (in ms) to register a valid speech segment. |
| `min_silence_ms`| `int` | `100` | Minimum continuous silence (in ms) required to successfully end a speech segment. |
| `hangover_ms` | `int` | `300` | Added padding (in ms) after speech trails off to prevent clipping word tails. |

---

## Testing

### Running the Test Suite

```bash
# Build the extension first
uv run maturin develop

# Run all tests
uv run --extra test pytest tests/test_vad.py -v
```

### Test Coverage

The test suite (`tests/test_vad.py`) covers all three models:

**Energy model tests** — basic pipeline validation using simple sine waves:
- Detector initialization
- Silence detection (low confidence)
- Speech detection (loud tone triggers above threshold)
- Stream processing yields properly-bounded `SpeechSegment` objects

**Silero VAD tests** (`TestSilero`):
- Initialization at 16kHz and 8kHz
- Silence produces low confidence (<0.3)
- Speech-like harmonics detected at 8kHz (native mode, >0.3 confidence)
- Reset clears internal RNN state
- Stream processing detects speech segments

**TEN VAD tests** (`TestTenVad`):
- Initialization at 16kHz and 8kHz (auto-resampling)
- Silence produces low confidence (<0.3)
- Speech-like harmonics detected at 16kHz (>0.3 confidence)
- Reset clears RNN state and feature buffers
- Stream processing detects speech segments

### Synthetic Test Signal

Tests use a `generate_speech_harmonics()` helper that produces a more realistic synthetic signal than a pure sine wave:
- Fundamental at 150 Hz with natural harmonic roll-off (1/n weighting)
- Amplitude modulation at ~6 Hz (syllable rate) for speech-like envelope
- Light noise (~5%) simulating glottal turbulence
- Deterministic seed (`random.seed(42)`) for reproducibility

This signal is sufficient for TEN VAD testing but Silero requires a lower threshold on synthetic audio due to its strict speech-specific training. With real microphone/telephony audio, both models work well at standard thresholds (0.5).