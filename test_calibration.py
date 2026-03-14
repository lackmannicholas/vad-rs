import math, struct, audioop
from vad_rs import VoiceActivityDetector, AudioFormat


def sine_pcm16(freq, dur_ms, sr, amp=0.5):
    n = int(sr * (dur_ms / 1000.0))
    return b"".join(struct.pack("<h", int(amp * math.sin(2 * math.pi * freq * i / sr) * 32767)) for i in range(n))


def pcm16_to_ulaw(pcm_bytes):
    return audioop.lin2ulaw(pcm_bytes, 2)


# Simulate Twilio: 8kHz µ-law, 160-byte packets
# Test with realistic telephony amplitudes (0.05-0.15 for normal speech)

print("=== Energy model (8kHz µ-law) ===")
for amp, label in [(0.005, "background noise"), (0.03, "quiet speech"), (0.1, "normal speech"), (0.2, "loud speech"), (0.5, "yelling")]:
    vad = VoiceActivityDetector(model="energy", input_sample_rate=8000, input_format=AudioFormat.Ulaw, threshold=0.5, min_speech_ms=20)
    tone_pcm = sine_pcm16(300.0, 200, 8000, amp)
    tone_ulaw = pcm16_to_ulaw(tone_pcm)
    confs = []
    for i in range(0, len(tone_ulaw), 160):
        r = vad.process_frame(tone_ulaw[i : i + 160])
        if r.confidence > 0:
            confs.append(r.confidence)
    avg = sum(confs) / len(confs) if confs else 0
    print(f"  {label:20s} (amp={amp:.3f}): avg confidence={avg:.4f}")

print()
print("=== Silero model (8kHz µ-law, native 8kHz) ===")
for amp, label in [(0.005, "background noise"), (0.03, "quiet speech"), (0.1, "normal speech"), (0.2, "loud speech"), (0.5, "yelling")]:
    vad = VoiceActivityDetector(model="silero", input_sample_rate=8000, input_format=AudioFormat.Ulaw, threshold=0.5, min_speech_ms=20)
    tone_pcm = sine_pcm16(300.0, 500, 8000, amp)
    tone_ulaw = pcm16_to_ulaw(tone_pcm)
    confs = []
    for i in range(0, len(tone_ulaw), 160):
        r = vad.process_frame(tone_ulaw[i : i + 160])
        confs.append(r.confidence)
    avg = sum(confs) / len(confs) if confs else 0
    mx = max(confs) if confs else 0
    print(f"  {label:20s} (amp={amp:.3f}): avg={avg:.4f}, max={mx:.4f}")

print()
print("=== Silero 16kHz baseline (PCM16, no resampling) ===")
for amp, label in [(0.005, "background noise"), (0.03, "quiet speech"), (0.1, "normal speech"), (0.2, "loud speech"), (0.5, "yelling")]:
    vad = VoiceActivityDetector(model="silero", input_sample_rate=16000, input_format=AudioFormat.Pcm16, threshold=0.5, min_speech_ms=20)
    tone = sine_pcm16(300.0, 500, 16000, amp)
    confs = []
    chunk_size = 512 * 2  # 512 samples * 2 bytes
    for i in range(0, len(tone), chunk_size):
        r = vad.process_frame(tone[i : i + chunk_size])
        confs.append(r.confidence)
    avg = sum(confs) / len(confs) if confs else 0
    mx = max(confs) if confs else 0
    print(f"  {label:20s} (amp={amp:.3f}): avg={avg:.4f}, max={mx:.4f}")
