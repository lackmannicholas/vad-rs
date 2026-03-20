use super::VadModel;
use ort::session::Session;
use ort::value::Tensor;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants matching TEN VAD C source (aed_st.h / coeff.h)
// ---------------------------------------------------------------------------
const FS: usize = 16000;
const FFT_SIZE: usize = 1024;
const WINDOW_SIZE: usize = 768;
const HOP_SIZE: usize = 256; // model frame size
const N_BINS: usize = FFT_SIZE / 2 + 1; // 513
const MEL_BANDS: usize = 40;
const FEA_LEN: usize = MEL_BANDS + 1; // 41 (40 mel + 1 pitch)
const CONTEXT_LEN: usize = 3;
const HIDDEN_DIM: usize = 64;
const NUM_STATES: usize = 4; // input_2, input_3, input_6, input_7
const EPS: f32 = 1e-20;
const PRE_EMPH_COEFF: f32 = 0.97;

// Feature normalization arrays from coeff.h
const FEATURE_MEANS: [f32; FEA_LEN] = [
    -8.198236465454e+00,
    -6.265716552734e+00,
    -5.483818531036e+00,
    -4.758691310883e+00,
    -4.417088985443e+00,
    -4.142892837524e+00,
    -3.912850379944e+00,
    -3.845927953720e+00,
    -3.657090425491e+00,
    -3.723418712616e+00,
    -3.876134157181e+00,
    -3.843890905380e+00,
    -3.690405130386e+00,
    -3.756065845490e+00,
    -3.698696136475e+00,
    -3.650463104248e+00,
    -3.700468778610e+00,
    -3.567321300507e+00,
    -3.498900175095e+00,
    -3.477807044983e+00,
    -3.458816051483e+00,
    -3.444923877716e+00,
    -3.401328563690e+00,
    -3.306261301041e+00,
    -3.278556823730e+00,
    -3.233250856400e+00,
    -3.198616027832e+00,
    -3.204526424408e+00,
    -3.208798646927e+00,
    -3.257838010788e+00,
    -3.381376743317e+00,
    -3.534021377563e+00,
    -3.640867948532e+00,
    -3.726858854294e+00,
    -3.773730993271e+00,
    -3.804667234421e+00,
    -3.832901000977e+00,
    -3.871120452881e+00,
    -3.990592956543e+00,
    -4.480289459229e+00,
    9.235690307617e+01, // pitch mean ≈ 92.4 Hz
];

const FEATURE_STDS: [f32; FEA_LEN] = [
    5.166063785553e+00,
    4.977209568024e+00,
    4.698895931244e+00,
    4.630621433258e+00,
    4.634347915649e+00,
    4.641156196594e+00,
    4.640676498413e+00,
    4.666367053986e+00,
    4.650534629822e+00,
    4.640020847321e+00,
    4.637400150299e+00,
    4.620099067688e+00,
    4.596316337585e+00,
    4.562654972076e+00,
    4.554360389709e+00,
    4.566910743713e+00,
    4.562489986420e+00,
    4.562412738800e+00,
    4.585299491882e+00,
    4.600179672241e+00,
    4.592845916748e+00,
    4.585922718048e+00,
    4.583496570587e+00,
    4.626092910767e+00,
    4.626957893372e+00,
    4.626289367676e+00,
    4.637005805969e+00,
    4.683015823364e+00,
    4.726813793182e+00,
    4.734289646149e+00,
    4.753227233887e+00,
    4.849722862244e+00,
    4.869434833527e+00,
    4.884482860565e+00,
    4.921327114105e+00,
    4.959212303162e+00,
    4.996619224548e+00,
    5.044823646545e+00,
    5.072216987610e+00,
    5.096439361572e+00,
    1.152136917114e+02, // pitch std ≈ 115.2 Hz
];

// ---------------------------------------------------------------------------
// TEN VAD Model
// ---------------------------------------------------------------------------

pub struct TenVadModel {
    session: Session,
    /// RNN hidden states: 4 × [HIDDEN_DIM] (for input_2/3/6/7)
    states: [[f32; HIDDEN_DIM]; NUM_STATES],
    /// Hann window coefficients (768 points)
    window: Vec<f32>,
    /// STFT overlap buffer — original signal for pitch estimation
    overlap_buf: Vec<f32>,
    /// STFT overlap buffer — pre-emphasized signal for spectral analysis
    overlap_emph_buf: Vec<f32>,
    /// Previous sample for pre-emphasis filter
    pre_emph_prev: f32,
    /// Feature context stack: [CONTEXT_LEN * FEA_LEN] = 3 × 41 = 123 values
    feat_stack: Vec<f32>,
    /// Mel filterbank matrix: [MEL_BANDS][N_BINS] = [40][513]
    mel_filterbank: Vec<Vec<f32>>,
    /// Cached FFT plan for reuse
    fft: Arc<dyn Fft<f32>>,
}

impl TenVadModel {
    pub fn new(model_path: &str) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("Failed to create ORT session builder: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("Failed to load TEN VAD ONNX model from '{}': {}", model_path, e))?;

        // Generate Hann window: w[n] = sin²(πn/N), matching TEN VAD's Hann768
        let window: Vec<f32> = (0..WINDOW_SIZE)
            .map(|n| {
                let s = (PI * n as f32 / WINDOW_SIZE as f32).sin();
                s * s
            })
            .collect();

        let mel_filterbank = Self::compute_mel_filterbank();

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);

        Ok(Self {
            session,
            states: [[0.0f32; HIDDEN_DIM]; NUM_STATES],
            window,
            overlap_buf: vec![0.0; WINDOW_SIZE],
            overlap_emph_buf: vec![0.0; WINDOW_SIZE],
            pre_emph_prev: 0.0,
            feat_stack: vec![0.0; CONTEXT_LEN * FEA_LEN],
            mel_filterbank,
            fft,
        })
    }

    // -----------------------------------------------------------------------
    // Mel filterbank computation (matches TEN VAD C source)
    // -----------------------------------------------------------------------

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    fn compute_mel_filterbank() -> Vec<Vec<f32>> {
        let low_mel = Self::hz_to_mel(0.0);
        let high_mel = Self::hz_to_mel(8000.0);

        // Compute mel-spaced bin indices (matching C code exactly)
        let mut mel_bin = vec![0usize; MEL_BANDS + 2];
        for i in 0..MEL_BANDS + 2 {
            let mel_pt = i as f32 * (high_mel - low_mel) / (MEL_BANDS as f32 + 1.0) + low_mel;
            let hz_pt = Self::mel_to_hz(mel_pt);
            // C code: (size_t)((stHdl->intFftSz + 1.0f) * hz_points / AUP_AED_FS)
            mel_bin[i] = ((FFT_SIZE as f32 + 1.0) * hz_pt / FS as f32) as usize;
        }

        // Build triangular filterbank
        let mut fb = vec![vec![0.0f32; N_BINS]; MEL_BANDS];
        for j in 0..MEL_BANDS {
            // Rising slope
            if mel_bin[j + 1] > mel_bin[j] {
                for i in mel_bin[j]..mel_bin[j + 1] {
                    if i < N_BINS {
                        fb[j][i] = (i - mel_bin[j]) as f32 / (mel_bin[j + 1] - mel_bin[j]) as f32;
                    }
                }
            }
            // Falling slope
            if mel_bin[j + 2] > mel_bin[j + 1] {
                for i in mel_bin[j + 1]..mel_bin[j + 2] {
                    if i < N_BINS {
                        fb[j][i] =
                            (mel_bin[j + 2] - i) as f32 / (mel_bin[j + 2] - mel_bin[j + 1]) as f32;
                    }
                }
            }
        }
        fb
    }

    // -----------------------------------------------------------------------
    // Pre-emphasis filter
    // -----------------------------------------------------------------------

    fn apply_pre_emphasis(&mut self, samples: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(samples.len());
        for &s in samples {
            output.push(s - PRE_EMPH_COEFF * self.pre_emph_prev);
            self.pre_emph_prev = s;
        }
        output
    }

    // -----------------------------------------------------------------------
    // STFT → power spectrum
    // -----------------------------------------------------------------------

    fn compute_bin_power(&self, emph_buf: &[f32]) -> Vec<f32> {
        // Apply window and create complex FFT input (zero-padded to FFT_SIZE)
        let mut fft_buf = vec![Complex::new(0.0f32, 0.0f32); FFT_SIZE];
        for i in 0..WINDOW_SIZE {
            fft_buf[i] = Complex::new(emph_buf[i] * self.window[i], 0.0);
        }
        // remaining positions are zero (zero-padding 768→1024)

        self.fft.process(&mut fft_buf);

        // Magnitude spectrum for first N_BINS = 513 bins.
        // TEN VAD's mel filterbank is applied to the linear magnitude spectrum
        // (not the squared power spectrum). Using squared power shifts the
        // log-mel features ~11 units above the model's training distribution,
        // causing near-zero confidence output for all inputs.
        let mut bin_power = vec![0.0f32; N_BINS];
        for i in 0..N_BINS {
            bin_power[i] = (fft_buf[i].re * fft_buf[i].re + fft_buf[i].im * fft_buf[i].im).sqrt();
        }
        bin_power
    }

    // -----------------------------------------------------------------------
    // Mel filterbank features
    // -----------------------------------------------------------------------

    fn compute_mel_features(&self, bin_power: &[f32]) -> [f32; MEL_BANDS] {
        let power_normal: f32 = 32768.0 * 32768.0;
        let mut mel = [0.0f32; MEL_BANDS];

        for i in 0..MEL_BANDS {
            let mut band_val = 0.0f32;
            for j in 0..N_BINS {
                band_val += bin_power[j] * self.mel_filterbank[i][j];
            }
            band_val /= power_normal;
            mel[i] = (band_val + EPS).ln();
        }
        mel
    }

    // -----------------------------------------------------------------------
    // Pitch estimation (autocorrelation-based, simplified vs TEN VAD's LPCNet)
    // -----------------------------------------------------------------------

    fn estimate_pitch(signal: &[f32]) -> f32 {
        let len = signal.len();

        // Quick energy check
        let energy: f32 = signal.iter().map(|&s| s * s).sum();
        if energy < 1.0 {
            return 0.0;
        }

        // Search for pitch in 60–500 Hz range at 16 kHz
        let min_lag = FS / 500; // 32 samples
        let max_lag = std::cmp::min(FS / 60, len / 2); // 267 samples, capped

        let mut best_corr = 0.0f32;
        let mut best_lag = 0usize;

        for lag in min_lag..=max_lag {
            let n = len - lag;
            let mut corr = 0.0f32;
            let mut e1 = 0.0f32;
            let mut e2 = 0.0f32;
            for i in 0..n {
                corr += signal[i] * signal[i + lag];
                e1 += signal[i] * signal[i];
                e2 += signal[i + lag] * signal[i + lag];
            }
            let norm = (e1 * e2).sqrt();
            if norm > EPS {
                let nc = corr / norm;
                if nc > best_corr {
                    best_corr = nc;
                    best_lag = lag;
                }
            }
        }

        if best_corr > 0.3 && best_lag > 0 {
            FS as f32 / best_lag as f32
        } else {
            0.0
        }
    }

    // -----------------------------------------------------------------------
    // Full feature extraction pipeline (one hop = 256 samples)
    // -----------------------------------------------------------------------

    fn extract_features(&mut self, frame: &[f32]) {
        // Our pipeline delivers f32 in [-1.0, 1.0]. TEN VAD expects int16-scale.
        let scaled: Vec<f32> = frame.iter().map(|&s| s * 32768.0).collect();

        // --- Update overlap buffer for pitch (original signal) ---
        self.overlap_buf.copy_within(HOP_SIZE..WINDOW_SIZE, 0);
        self.overlap_buf[WINDOW_SIZE - HOP_SIZE..].copy_from_slice(&scaled);

        // --- Pre-emphasis ---
        let emph = self.apply_pre_emphasis(&scaled);

        // --- Update pre-emphasized overlap buffer for STFT ---
        self.overlap_emph_buf.copy_within(HOP_SIZE..WINDOW_SIZE, 0);
        self.overlap_emph_buf[WINDOW_SIZE - HOP_SIZE..].copy_from_slice(&emph);

        // --- STFT → power spectrum ---
        let bin_power = self.compute_bin_power(&self.overlap_emph_buf);

        // --- Mel filterbank ---
        let mel = self.compute_mel_features(&bin_power);

        // --- Pitch estimation ---
        let pitch = Self::estimate_pitch(&self.overlap_buf);

        // --- Normalize features ---
        let mut features = [0.0f32; FEA_LEN];
        for i in 0..MEL_BANDS {
            features[i] = (mel[i] - FEATURE_MEANS[i]) / (FEATURE_STDS[i] + EPS);
        }
        features[MEL_BANDS] = (pitch - FEATURE_MEANS[MEL_BANDS]) / (FEATURE_STDS[MEL_BANDS] + EPS);

        // --- Update context stack (shift oldest frame out, append newest) ---
        let total = CONTEXT_LEN * FEA_LEN;
        self.feat_stack.copy_within(FEA_LEN..total, 0);
        self.feat_stack[(CONTEXT_LEN - 1) * FEA_LEN..total].copy_from_slice(&features);
    }
}

// ---------------------------------------------------------------------------
// VadModel trait implementation
// ---------------------------------------------------------------------------

impl VadModel for TenVadModel {
    fn predict(&mut self, frame: &[f32]) -> f32 {
        // 1) Feature extraction (populates self.feat_stack)
        self.extract_features(frame);

        // 2) Build input_1 tensor: shape [1, 3, 41]
        let feat_tensor = match Tensor::from_array((
            [1_i64, CONTEXT_LEN as i64, FEA_LEN as i64],
            self.feat_stack.clone().into_boxed_slice(),
        )) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };

        // 3) Build state tensors: input_2/3/6/7, each [1, 64]
        let st2 = match Tensor::from_array((
            [1_i64, HIDDEN_DIM as i64],
            self.states[0].to_vec().into_boxed_slice(),
        )) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };
        let st3 = match Tensor::from_array((
            [1_i64, HIDDEN_DIM as i64],
            self.states[1].to_vec().into_boxed_slice(),
        )) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };
        let st6 = match Tensor::from_array((
            [1_i64, HIDDEN_DIM as i64],
            self.states[2].to_vec().into_boxed_slice(),
        )) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };
        let st7 = match Tensor::from_array((
            [1_i64, HIDDEN_DIM as i64],
            self.states[3].to_vec().into_boxed_slice(),
        )) {
            Ok(t) => t,
            Err(_) => return 0.0,
        };

        // 4) Run ONNX inference
        let outputs = match self.session.run(ort::inputs! {
            "input_1" => feat_tensor,
            "input_2" => st2,
            "input_3" => st3,
            "input_6" => st6,
            "input_7" => st7,
        }) {
            Ok(o) => o,
            Err(_) => return 0.0,
        };

        // 5) Extract speech probability from output_1 (shape [1, 1, 1])
        let confidence = match outputs["output_1"].try_extract_tensor::<f32>() {
            Ok((_shape, data)) => data[0],
            Err(_) => return 0.0,
        };

        // 6) Update RNN hidden states from output_2/3/6/7
        let state_output_names = ["output_2", "output_3", "output_6", "output_7"];
        for (i, name) in state_output_names.iter().enumerate() {
            if let Ok((_shape, data)) = outputs[*name].try_extract_tensor::<f32>() {
                let len = data.len().min(HIDDEN_DIM);
                self.states[i][..len].copy_from_slice(&data[..len]);
            }
        }

        confidence
    }

    fn reset(&mut self) {
        self.states = [[0.0f32; HIDDEN_DIM]; NUM_STATES];
        self.overlap_buf = vec![0.0; WINDOW_SIZE];
        self.overlap_emph_buf = vec![0.0; WINDOW_SIZE];
        self.pre_emph_prev = 0.0;
        self.feat_stack = vec![0.0; CONTEXT_LEN * FEA_LEN];
    }

    fn sample_rate(&self) -> u32 {
        FS as u32 // TEN VAD always operates at 16 kHz
    }

    fn frame_size(&self) -> usize {
        HOP_SIZE // 256 samples = 16 ms at 16 kHz
    }
}
