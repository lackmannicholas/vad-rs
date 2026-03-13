use crate::AudioFormat;

pub fn decode_audio(bytes: &[u8], format: &AudioFormat) -> Result<Vec<f32>, String> {
    match format {
        AudioFormat::Pcm16 => {
            if bytes.len() % 2 != 0 {
                return Err("PCM16 byte length must be even".to_string());
            }
            let samples: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    // Normalize to [-1.0, 1.0]
                    (sample as f32) / 32_768.0
                })
                .collect();
            Ok(samples)
        }
        AudioFormat::PcmF32 => {
            if bytes.len() % 4 != 0 {
                return Err("F32 byte length must be multiple of 4".to_string());
            }
            let samples: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Ok(samples)
        }
        AudioFormat::Ulaw => {
            let samples: Vec<f32> = bytes
                .iter()
                .map(|&byte| {
                    let val = !byte;
                    let sign = (val & 0x80) != 0;
                    let exponent = ((val >> 4) & 0x07) as i16;
                    let mantissa = (val & 0x0F) as i16;
                    let sample = (((mantissa << 3) + 132) << exponent) - 132;
                    let pcm = if sign { -sample } else { sample };
                    (pcm as f32) / 32_768.0
                })
                .collect();
            Ok(samples)
        }
        AudioFormat::Alaw => Err("alaw not implemented yet".to_string()),
    }
}
