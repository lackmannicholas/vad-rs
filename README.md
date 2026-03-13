# vad-rs

**vad-rs** is a production-grade Voice Activity Detection (VAD) library implemented in Rust with a clean, high-performance Python API. 

The Rust core handles audio preprocessing, frame management, audio resampling, and model inference, while Python consumers get an easy-to-use, native extension via PyO3 and Maturin.

## Features

- **Blazing Fast**: Core logic, state machines, and formatting run in Rust.
- **Multiple Models**: 
  - `silero`: (Default) Uses the ONNX runtime to run the highly accurate Silero VAD model.
  - `energy`: A lightweight RMS/Zero-crossing fallback that requires no ML dependencies.
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
python -m venv venv
source venv/bin/activate

# Install build dependencies
pip install maturin

# Build and install the Python bindings into your current environment
maturin develop --release

# Alternatively, if you are importing it into another project using pip:
# pip install -e /path/to/vad-rs
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
    model="silero",                      # "silero" or "energy"
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
    model="silero",
    input_sample_rate=16000,
    input_format=AudioFormat.Pcm16
)

# frame_iterator can be any iterable that yields `bytes`
# e.g., a generator reading from a .wav file or a network socket
def read_frames(filepath):
    chunk_size = 512 # adjust based on your needs
    with open(filepath, 'rb') as f:
        while chunk = f.read(chunk_size):
            yield chunk

for segment in vad.process_stream(read_frames("audio.wav")):
    print(
        f"Speech Segment: {segment.start_ms}ms to {segment.end_ms}ms "
        f"(Average Confidence: {segment.confidence:.2f})"
    )
```

## Configuration Parameters

The `VoiceActivityDetector` takes several tuning parameters in its constructor to adapt to your environment:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"silero"` | The backend model to use. Choose `"silero"` or `"energy"`. |
| `input_sample_rate` | `int` | `16000` | The sample rate of the input audio in Hz (e.g., `8000`, `16000`, `48000`). |
| `input_format` | `AudioFormat` | `AudioFormat.Pcm16` | The payload format (`AudioFormat.Pcm16`, `AudioFormat.PcmF32`, `AudioFormat.Ulaw`). |
| `threshold` | `float` | `0.5` | The probability threshold (0.0 - 1.0) above which a frame represents speech. |
| `min_speech_ms` | `int` | `250` | Minimum continuous duration (in ms) to register a valid speech segment. |
| `min_silence_ms`| `int` | `100` | Minimum continuous silence (in ms) required to successfully end a speech segment. |
| `hangover_ms` | `int` | `300` | Added padding (in ms) after speech trails off to prevent clipping word tails. |