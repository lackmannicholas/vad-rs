# vad-rs: Voice Activity Detection with Rust Core + Python Bindings

## Project Overview

A production-grade Voice Activity Detection (VAD) library implemented in Rust with a clean Python API via PyO3/maturin. The Rust core handles audio preprocessing, frame management, and model inference. Python consumers get a `pip install`-able package with native performance.

**Dual goals:**
1. Build a production-quality VAD suitable for real-time voice pipelines (telephony, WebRTC, etc.)
2. Learn the Rust → Python native extension packaging pattern (PyO3 + maturin)

---

## Architecture

### High-Level Design

```
Python API (vad_rs)
    │
    ▼
PyO3 Bindings (lib.rs)
    │
    ▼
┌─────────────────────────────────────────┐
│  Rust Core                              │
│                                         │
│  ┌─────────┐   ┌───────────┐           │
│  │ pipeline │──▶│ detector  │           │
│  │ resample │   │ state     │           │
│  │ framing  │   │ machine   │           │
│  │ format   │   │ hangover  │           │
│  └─────────┘   └─────┬─────┘           │
│                       │                  │
│               ┌───────▼────────┐        │
│               │  inference/    │        │
│               │  mod.rs (trait)│        │
│               │  onnx.rs      │        │
│               │  energy.rs    │        │
│               └────────────────┘        │
└─────────────────────────────────────────┘
```

### Project Structure

```
vad-rs/
├── Cargo.toml                  # Rust crate config
├── pyproject.toml              # maturin build config
├── src/
│   ├── lib.rs                  # PyO3 module entry point, Python-facing structs
│   ├── detector.rs             # Main VAD struct, state machine, hangover logic
│   ├── pipeline.rs             # Audio preprocessing: resampling, framing, feature extraction
│   ├── audio.rs                # Sample format conversion, ring buffer, frame slicing
│   ├── inference/
│   │   ├── mod.rs              # VadModel trait definition
│   │   ├── onnx.rs             # ONNX Runtime backend (Silero VAD, future custom models)
│   │   └── energy.rs           # Simple energy-based fallback (no model dependency)
├── models/
│   └── silero_vad.onnx         # Pretrained model (bundled or fetched at build)
├── python/
│   └── vad_rs/
│       ├── __init__.py         # Public Python API, re-exports from _core
│       └── _core.pyi           # Type stubs for IDE/mypy support
├── tests/
│   ├── test_vad.py             # Python-level integration tests
│   └── test_audio/             # Sample audio files for testing
└── examples/
    └── detect_speech.py        # Demo: detect speech segments in a WAV file
```

---

## Core Design: VadModel Trait

The inference backend is abstracted behind a trait, enabling model swapping without changing the pipeline or Python API.

```rust
pub trait VadModel: Send {
    /// Run inference on a single audio frame.
    /// Returns speech probability in range [0.0, 1.0].
    fn predict(&mut self, frame: &AudioFrame) -> f32;

    /// Reset internal model state (e.g., between calls/sessions).
    fn reset(&mut self);

    /// The sample rate the model expects.
    fn sample_rate(&self) -> u32;

    /// The frame size (in samples) the model expects.
    fn frame_size(&self) -> usize;
}
```

**Implementations:**
- `energy.rs` — RMS energy + zero-crossing rate. No external dependencies. Useful as a fallback and for validating the pipeline end-to-end before adding ONNX complexity.
- `onnx.rs` — ONNX Runtime inference. Ships with Silero VAD as the default model. Designed so a future custom-trained model drops in as a different `.onnx` file with the same interface.

---

## Audio Format Support

### Supported Input Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| PCM 16-bit signed (`pcm_16`) | Standard WAV format | General purpose |
| PCM 32-bit float (`pcm_f32`) | ML pipeline native format | Inter-process audio |
| μ-law (`ulaw`) | G.711 μ-law encoding | PSTN / North American telephony |
| A-law (`alaw`) | G.711 A-law encoding | PSTN / European telephony |

### Supported Sample Rates

| Rate | Context |
|------|---------|
| 8,000 Hz | Telephony (PSTN, G.711) — primary production use case |
| 16,000 Hz | Speech processing standard, Silero's native rate |
| 48,000 Hz | WebRTC native, high-quality capture |

The pipeline handles resampling internally. Input at any supported rate is resampled to the model's expected rate (16kHz for Silero) before inference. Resampling is handled by the `rubato` crate.

### Channel Handling

- Accept mono or stereo input
- Stereo is downmixed to mono before processing
- No multi-channel / surround support needed

---

## Python API

### Construction

```python
from vad_rs import VoiceActivityDetector, AudioFormat

vad = VoiceActivityDetector(
    model="silero",              # "silero" | "energy"
    input_sample_rate=8000,      # 8000 | 16000 | 48000
    input_format="pcm_16",       # "pcm_16" | "pcm_f32" | "ulaw" | "alaw"
    threshold=0.5,               # Speech probability threshold
    min_speech_ms=250,           # Min duration to count as speech
    min_silence_ms=100,          # Min silence before ending a segment
    hangover_ms=300,             # Extra padding after speech ends
)
```

### Frame-by-Frame Processing (Real-Time)

```python
# Process a single audio frame (bytes)
result = vad.process_frame(audio_bytes)

result.is_speech       # bool — is this frame speech?
result.confidence      # float — speech probability [0.0, 1.0]
```

### Streaming Segment Detection

```python
# Iterate over detected speech segments
for segment in vad.process_stream(frame_iterator):
    segment.start_ms    # int — segment start in milliseconds
    segment.end_ms      # int — segment end in milliseconds
    segment.confidence  # float — average confidence across segment
```

### State Management

```python
# Reset between calls/sessions (clears model state + ring buffer)
vad.reset()
```

---

## Detector State Machine

The detector manages transitions between speech and silence using the configured timing parameters:

```
                    ┌──────────────────┐
                    │                  │
         ┌─────────▼──────────┐       │
         │     SILENCE        │       │
         │                    │       │ confidence < threshold
         └─────────┬──────────┘       │ for min_silence_ms
                   │                  │
                   │ confidence       │
                   │ >= threshold     │
                   │                  │
         ┌─────────▼──────────┐       │
         │   PENDING_SPEECH   │       │
         │  (accumulating     │       │
         │   min_speech_ms)   │       │
         └─────────┬──────────┘       │
                   │                  │
                   │ duration >=      │
                   │ min_speech_ms    │
                   │                  │
         ┌─────────▼──────────┐       │
         │     SPEECH         ├───────┘
         │                    │
         │  (hangover_ms pad  │
         │   after last       │
         │   speech frame)    │
         └────────────────────┘
```

**Parameters:**
- `threshold` — speech probability cutoff (0.0–1.0)
- `min_speech_ms` — debounce: ignore speech shorter than this
- `min_silence_ms` — debounce: ignore silence gaps shorter than this
- `hangover_ms` — extend speech segments by this much after the last above-threshold frame

---

## Rust Crate Dependencies

| Crate | Purpose |
|-------|---------|
| `pyo3` | Python bindings (with `extension-module` feature) |
| `numpy` (pyo3-numpy) | Zero-copy buffer passing between Python and Rust |
| `ort` | ONNX Runtime bindings (maintained fork, replaces `onnxruntime-rs`) |
| `rubato` | High-quality async sample rate conversion |
| `rustfft` | FFT for spectral features (if added later) |
| `hound` | WAV file reading for tests/examples |

---

## Build Configuration

### pyproject.toml (maturin)

```toml
[build-system]
requires = ["maturin>=1.0,<2.0"]
build-backend = "maturin"

[project]
name = "vad-rs"
requires-python = ">=3.9"

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "python"
module-name = "vad_rs._core"
```

### Development Workflow

```bash
# Setup
pip install maturin

# Build and install in dev mode (rebuilds Rust on change)
maturin develop

# Run Python tests
pytest tests/

# Run Rust tests
cargo test

# Build wheel for distribution
maturin build --release
```

---

## Implementation Order

Build incrementally, validating each layer before adding the next:

### Phase 1: Audio Primitives
**Files:** `audio.rs`, `pipeline.rs`
- Sample format conversion (pcm_16 ↔ f32, ulaw/alaw decode)
- Ring buffer for frame accumulation
- Frame slicing at configurable durations
- Stereo → mono downmix
- **Test:** Pure Rust unit tests, feed known audio, verify output samples

### Phase 2: Energy-Based Detector (End-to-End Validation)
**Files:** `inference/mod.rs`, `inference/energy.rs`, `detector.rs`, `lib.rs`
- Define `VadModel` trait
- Implement energy-based backend (RMS + zero-crossing)
- Build detector state machine with hangover logic
- Wire up PyO3 bindings: `VoiceActivityDetector` class, `process_frame`, `reset`
- **Test:** Python integration test — create detector, feed audio frames, get results
- **Validates:** Trait design, Python bindings, frame pipeline, state machine — all without ONNX

### Phase 3: ONNX / Silero Integration
**Files:** `inference/onnx.rs`, resampling additions to `pipeline.rs`
- Integrate `ort` crate, load Silero ONNX model
- Implement resampling via `rubato` (8kHz/48kHz → 16kHz for Silero)
- Implement `VadModel` trait for ONNX backend
- **Test:** Compare output against reference Silero Python implementation on same audio

### Phase 4: Streaming API
**Files:** Updates to `lib.rs`, `detector.rs`
- `process_stream` — Python iterator protocol over speech segments
- Segment boundary detection using state machine transitions
- Segment metadata: start_ms, end_ms, average confidence
- **Test:** Feed known audio with speech/silence regions, verify segment boundaries

### Phase 5: Polish
**Files:** `python/vad_rs/__init__.py`, `python/vad_rs/_core.pyi`
- Clean Python re-exports in `__init__.py`
- Type stubs (`.pyi`) for IDE autocomplete and mypy
- Error handling: clear Python exceptions for bad input formats, missing models, etc.
- Docstrings on all public Python-facing methods

---

## Future Work (Out of Scope for v1)

- **Custom model training** — train a small VAD model on telephony audio (8kHz, codec artifacts, office noise), export to ONNX, drop into the same `VadModel` trait
- **PyPI publishing** — `maturin publish`, CI/CD for cross-platform wheels
- **WebAssembly target** — compile Rust core to WASM for browser-native VAD
- **Spectral features** — add FFT-based features (spectral flatness, entropy) as an intermediate backend between energy and full ML
- **Benchmarking** — latency per frame, memory usage, comparison against pure-Python Silero wrapper