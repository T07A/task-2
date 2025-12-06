"""
ASR Audio Processing Pipeline - Implemented from Scratch
========================================================
This script demonstrates the fundamental audio processing steps for ASR
WITHOUT using high-level libraries like librosa or noisereduce.

Only allowed: numpy and scipy.io.wavfile (for loading only)

Author: DSP Engineering Student
Purpose: Understanding the mathematical foundations of audio processing
"""

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_FILE = "input.wav"  # Your audio file
OUTPUT_FILE = "final_processed.wav"

# DSP Parameters
TARGET_SAMPLE_RATE = 16000  # Target for ASR (will be approximated)
DECIMATION_FACTOR = 3       # For 44.1kHz → ~14.7kHz downsampling
FRAME_SIZE_MS = 25          # Frame size in milliseconds
FRAME_STRIDE_MS = 10        # Frame stride in milliseconds
PRE_EMPHASIS_COEF = 0.97
SILENCE_THRESHOLD_RATIO = 0.02  # 2% of mean energy (very conservative)
MOVING_AVG_WINDOW = 5       # For noise reduction

print("=" * 80)
print("ASR AUDIO PROCESSING PIPELINE - FROM SCRATCH IMPLEMENTATION")
print("=" * 80)

# =============================================================================
# STEP 1: ANALOG-TO-DIGITAL CONVERSION (ADC) SIMULATION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: ANALOG-TO-DIGITAL CONVERSION (ADC)")
print("=" * 80)

# Load the WAV file using scipy (simulates the digital capture from ADC)
original_sample_rate, audio_data = wavfile.read(INPUT_FILE)

print(f"\n[1.1] File Loaded: {INPUT_FILE}")
print(f"      Original Sampling Rate: {original_sample_rate} Hz")
print(f"      Data Type: {audio_data.dtype}")
print(f"      Shape: {audio_data.shape}")

# =============================================================================
# STEP 1.2: MANUAL STEREO TO MONO CONVERSION
# =============================================================================
print(f"\n[1.2] STEREO TO MONO CONVERSION")

if audio_data.ndim == 2:
    print(f"      ✓ STEREO DETECTED ({audio_data.shape[1]} channels)")
    print(f"      Original Shape: {audio_data.shape} (samples, channels)")
    print(f"      Method: Manual averaging across channels")
    print(f"      Formula: mono[n] = mean(all_channels[n])")
    
    # Manual averaging: take mean across columns (axis=1)
    audio_mono = audio_data.mean(axis=1).astype(np.float32)
    print(f"      New Shape: {audio_mono.shape}")
    print(f"      ✓ Converted to MONO")
else:
    print(f"      ✓ Already MONO - No conversion needed")
    audio_mono = audio_data.astype(np.float32)

print(f"\n[1.3] Original Duration: {len(audio_mono) / original_sample_rate:.2f} seconds")
print(f"      Total Samples: {len(audio_mono):,}")

# =============================================================================
# STEP 1.4: MANUAL DOWNSAMPLING (DECIMATION)
# =============================================================================
print(f"\n[1.4] MANUAL DOWNSAMPLING (DECIMATION)")
print(f"      Original Sample Rate: {original_sample_rate} Hz")
print(f"      Target for ASR: ~{TARGET_SAMPLE_RATE} Hz")

if original_sample_rate > TARGET_SAMPLE_RATE * 1.5:
    print(f"\n      [MATH CONCEPT - DECIMATION]:")
    print(f"      • Method: Take every Nth sample (downsampling)")
    print(f"      • Formula: decimated[i] = original[i * N] for i=0,1,2...")
    print(f"      • Decimation Factor: {DECIMATION_FACTOR}")
    print(f"      • New Rate: {original_sample_rate} / {DECIMATION_FACTOR} = {original_sample_rate // DECIMATION_FACTOR} Hz")
    
    # Manual decimation: take every 3rd sample
    # This is a simple downsampling (no anti-aliasing filter for educational purposes)
    audio_downsampled = audio_mono[::DECIMATION_FACTOR]
    sample_rate = original_sample_rate // DECIMATION_FACTOR
    
    print(f"\n      Before: {len(audio_mono):,} samples at {original_sample_rate} Hz")
    print(f"      After:  {len(audio_downsampled):,} samples at {sample_rate} Hz")
    print(f"      ✓ Data reduced by {DECIMATION_FACTOR}x")
else:
    print(f"      ✓ Sample rate already close to target - No decimation needed")
    audio_downsampled = audio_mono
    sample_rate = original_sample_rate

# Calculate frame size and stride based on NEW sample rate
FRAME_SIZE = int(FRAME_SIZE_MS * sample_rate / 1000)
FRAME_STRIDE = int(FRAME_STRIDE_MS * sample_rate / 1000)

print(f"\n[1.5] DSP Parameters (Adjusted for {sample_rate} Hz):")
print(f"      Frame Size: {FRAME_SIZE} samples ({FRAME_SIZE_MS}ms)")
print(f"      Frame Stride: {FRAME_STRIDE} samples ({FRAME_STRIDE_MS}ms)")

# =============================================================================
# STEP 1.6: MANUAL NORMALIZATION TO FLOAT [-1, 1]
# =============================================================================
print(f"\n[1.6] MANUAL NORMALIZATION TO FLOAT [-1, 1]:")

print(f"      Current data type: {audio_downsampled.dtype}")
max_abs_value = np.max(np.abs(audio_downsampled))
print(f"      Max absolute value: {max_abs_value:.2f}")

print(f"      Formula: normalized = audio / max_absolute_value")

# Normalize to [-1, 1]
if max_abs_value > 0:
    audio_float = audio_downsampled / max_abs_value
else:
    print(f"      WARNING: Audio is silent (all zeros)")
    audio_float = audio_downsampled

print(f"      New Range: [{audio_float.min():.6f}, {audio_float.max():.6f}]")
print(f"      First 5 samples: {audio_float[:5]}")
print(f"      ✓ Normalized to float [-1, 1]")

# =============================================================================
# STEP 2: PREPROCESSING - NOISE REDUCTION (MOVING AVERAGE FILTER)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2.1: NOISE REDUCTION - MOVING AVERAGE FILTER")
print("=" * 80)

print(f"\n[2.1] Applying Moving Average Filter (Window Size: {MOVING_AVG_WINDOW})")
print(f"\n[MATH CONCEPT - MOVING AVERAGE]:")
print(f"  • Formula: y[n] = (1/M) * Σ x[n-k] for k=0 to M-1")
print(f"  • M = {MOVING_AVG_WINDOW} (window size)")
print(f"  • Effect: Smooths the signal by averaging neighboring samples")
print(f"  • Removes high-frequency noise (acts as low-pass filter)")

# Manual Implementation using numpy convolution
# This is equivalent to: y[i] = (x[i-2] + x[i-1] + x[i] + x[i+1] + x[i+2]) / 5
kernel = np.ones(MOVING_AVG_WINDOW) / MOVING_AVG_WINDOW
audio_denoised = np.convolve(audio_float, kernel, mode='same')

print(f"\n[2.1.1] Noise Reduction Complete")
print(f"        Before: RMS = {np.sqrt(np.mean(audio_float**2)):.6f}")
print(f"        After:  RMS = {np.sqrt(np.mean(audio_denoised**2)):.6f}")

# =============================================================================
# STEP 2.2: PREPROCESSING - SILENCE REMOVAL (SHORT-TIME ENERGY)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2.2: SILENCE REMOVAL - SHORT-TIME ENERGY (STE)")
print("=" * 80)

print(f"\n[MATH CONCEPT - SHORT-TIME ENERGY]:")
print(f"  • Formula: E = Σ x[n]² (sum of squared samples in a frame)")
print(f"  • Frame Size: {FRAME_SIZE} samples ({FRAME_SIZE/sample_rate*1000:.1f} ms)")
print(f"  • Dynamic Threshold: {SILENCE_THRESHOLD_RATIO*100}% of mean energy")
print(f"  • Logic: If E < threshold → consider it silence, remove it")

def compute_short_time_energy(signal, frame_size):
    """
    Manually compute Short-Time Energy for silence detection
    
    Math: For each frame of length N:
          E = (1/N) * Σ(x[i]²) for i in frame
    """
    num_frames = len(signal) // frame_size
    energy = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * frame_size
        end = start + frame_size
        if end <= len(signal):
            frame = signal[start:end]
            # Energy calculation: sum of squares, normalized by frame length
            energy[i] = np.sum(frame ** 2) / frame_size
    
    return energy

def remove_silence_dynamic(signal, frame_size, threshold_ratio):
    """
    Remove silence frames based on Short-Time Energy with DYNAMIC threshold
    
    Key Fix: Uses mean energy instead of hardcoded threshold
    This prevents cutting off soft speech
    """
    num_frames = len(signal) // frame_size
    energy = compute_short_time_energy(signal, frame_size)
    
    # Calculate DYNAMIC threshold based on MEAN energy (not max!)
    mean_energy = np.mean(energy)
    threshold = threshold_ratio * mean_energy
    
    print(f"\n[2.2.1] Computing Short-Time Energy (Dynamic Threshold)...")
    print(f"        Total Frames: {num_frames}")
    print(f"        Mean Energy: {mean_energy:.8f}")
    print(f"        Threshold: {threshold:.8f} ({threshold_ratio*100}% of mean)")
    
    # Keep frames where energy > threshold
    voice_frames = []
    removed_frames = 0
    
    for i in range(num_frames):
        if energy[i] > threshold:
            start = i * frame_size
            end = start + frame_size
            if end <= len(signal):
                voice_frames.append(signal[start:end])
        else:
            removed_frames += 1
    
    if len(voice_frames) == 0:
        print(f"        ⚠️  WARNING: All frames below threshold! Keeping original.")
        return signal, 0
    
    print(f"        Frames Removed: {removed_frames} / {num_frames} ({removed_frames/num_frames*100:.1f}%)")
    
    return np.concatenate(voice_frames), removed_frames

audio_no_silence, removed_count = remove_silence_dynamic(audio_denoised, FRAME_SIZE, SILENCE_THRESHOLD_RATIO)

print(f"        Original Frames: {len(audio_denoised) // FRAME_SIZE}")
print(f"        Remaining Frames: {len(audio_no_silence) // FRAME_SIZE}")
print(f"        Time Removed: {(removed_count * FRAME_SIZE) / sample_rate:.2f} seconds")

# =============================================================================
# STEP 3: FEATURE EXTRACTION - PRE-EMPHASIS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3.1: FEATURE EXTRACTION - PRE-EMPHASIS FILTER")
print("=" * 80)

print(f"\n[MATH CONCEPT - PRE-EMPHASIS]:")
print(f"  • Formula: y[n] = x[n] - α * x[n-1]")
print(f"  • α (alpha) = {PRE_EMPHASIS_COEF}")
print(f"  • Purpose: Amplify high frequencies (balances frequency spectrum)")
print(f"  • Reason: Human speech has more energy in low frequencies")
print(f"  • Effect: Makes all frequencies more equally represented")

def apply_preemphasis(signal, coef=0.97):
    """
    Manually apply pre-emphasis filter
    
    Math: y[n] = x[n] - α * x[n-1]
    This is a first-order high-pass filter
    """
    emphasized = np.zeros_like(signal)
    emphasized[0] = signal[0]  # First sample unchanged
    
    # Apply the filter manually
    for n in range(1, len(signal)):
        emphasized[n] = signal[n] - coef * signal[n-1]
    
    return emphasized

audio_emphasized = apply_preemphasis(audio_no_silence, PRE_EMPHASIS_COEF)

print(f"\n[3.1.1] Pre-emphasis Applied")
print(f"        Formula used: y[n] = x[n] - {PRE_EMPHASIS_COEF} * x[n-1]")
print(f"        Before (first 5): {audio_no_silence[:5]}")
print(f"        After  (first 5): {audio_emphasized[:5]}")

# =============================================================================
# STEP 3.2: FRAMING AND WINDOWING
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3.2: FRAMING AND WINDOWING (HAMMING WINDOW)")
print("=" * 80)

print(f"\n[MATH CONCEPT - FRAMING]:")
print(f"  • Frame Size: {FRAME_SIZE} samples ({FRAME_SIZE/sample_rate*1000:.1f} ms)")
print(f"  • Frame Stride: {FRAME_STRIDE} samples ({FRAME_STRIDE/sample_rate*1000:.1f} ms)")
print(f"  • Overlap: {FRAME_SIZE - FRAME_STRIDE} samples")
print(f"  • Purpose: Break long signal into short, overlapping segments")

print(f"\n[MATH CONCEPT - HAMMING WINDOW]:")
print(f"  • Formula: w[n] = 0.54 - 0.46 * cos(2π * n / (N-1))")
print(f"  • N = {FRAME_SIZE} (frame length)")
print(f"  • Purpose: Reduce spectral leakage (smooth frame edges to zero)")
print(f"  • Effect: Multiplies each frame by this bell-shaped window")

def manual_hamming_window(frame_length):
    """
    Manually compute Hamming window
    
    Math: w[n] = 0.54 - 0.46 * cos(2πn / (N-1))
    where n = 0, 1, 2, ..., N-1
    """
    n = np.arange(frame_length)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (frame_length - 1))
    return window

def frame_signal(signal, frame_size, frame_stride):
    """
    Manually divide signal into overlapping frames
    """
    signal_length = len(signal)
    num_frames = 1 + int((signal_length - frame_size) / frame_stride)
    
    frames = np.zeros((num_frames, frame_size))
    
    for i in range(num_frames):
        start = i * frame_stride
        end = start + frame_size
        if end <= signal_length:
            frames[i] = signal[start:end]
        else:
            # Pad with zeros if needed
            remaining = signal[start:]
            frames[i, :len(remaining)] = remaining
    
    return frames

# Create Hamming window manually
hamming = manual_hamming_window(FRAME_SIZE)
print(f"\n[3.2.1] Hamming Window Generated")
print(f"        First 5 values: {hamming[:5]}")
print(f"        Middle value: {hamming[FRAME_SIZE//2]:.4f}")
print(f"        Last 5 values: {hamming[-5:]}")

# Frame the signal
frames = frame_signal(audio_emphasized, FRAME_SIZE, FRAME_STRIDE)
print(f"\n[3.2.2] Signal Framed")
print(f"        Total Frames: {len(frames)}")
print(f"        Each Frame: {FRAME_SIZE} samples")

# Apply window to each frame (element-wise multiplication)
windowed_frames = frames * hamming

print(f"\n[3.2.3] Windowing Applied")
print(f"        Each frame multiplied by Hamming window")
print(f"        Frame 0 before window (first 5): {frames[0, :5]}")
print(f"        Frame 0 after window  (first 5): {windowed_frames[0, :5]}")

# =============================================================================
# STEP 3.3: COMPUTE FFT (POWER SPECTRUM)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3.3: FAST FOURIER TRANSFORM (FFT) - POWER SPECTRUM")
print("=" * 80)

print(f"\n[MATH CONCEPT - FFT]:")
print(f"  • Converts time-domain signal → frequency-domain")
print(f"  • Formula (DFT): X[k] = Σ x[n] * e^(-j*2π*k*n/N)")
print(f"  • FFT is fast algorithm to compute DFT")
print(f"  • Output: Complex numbers representing amplitude & phase")
print(f"  • Power Spectrum: |X[k]|² (magnitude squared)")

# Compute FFT for each frame
fft_frames = np.fft.rfft(windowed_frames, FRAME_SIZE)
power_spectrum = np.abs(fft_frames) ** 2

print(f"\n[3.3.1] FFT Computed")
print(f"        FFT Size: {FRAME_SIZE}")
print(f"        Frequency Bins: {fft_frames.shape[1]}")
print(f"        Frequency Resolution: {sample_rate / FRAME_SIZE:.2f} Hz/bin")

print(f"\n[3.3.2] Power Spectrum Calculated")
print(f"        Formula: P[k] = |FFT[k]|²")
print(f"        Shape: {power_spectrum.shape}")
print(f"        Frame 0 Power (first 5 bins): {power_spectrum[0, :5]}")

# Show frequency bins
freq_bins = np.fft.rfftfreq(FRAME_SIZE, 1/sample_rate)
print(f"\n[3.3.3] Frequency Analysis")
print(f"        Frequency range: 0 Hz to {freq_bins[-1]:.1f} Hz")
print(f"        First 5 frequency bins (Hz): {freq_bins[:5]}")

# =============================================================================
# STEP 3.4: MEL FILTER BANKS (MFCC EXTENSION)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3.4: MEL FILTER BANKS")
print("=" * 80)

print(f"\n[MATH CONCEPT - MEL SCALE]:")
print(f"  • Formula: Mel(f) = 2595 × log10(1 + f/700)")
print(f"  • Purpose: Convert linear frequency to perceptual Mel scale")
print(f"  • Reason: Human hearing is more sensitive to changes at low frequencies")
print(f"  • Filter Banks: 40 triangular filters spanning frequency range")

# Parameters for Mel Filter Banks
NUM_MEL_FILTERS = 40
NUM_MFCC_COEFFS = 13

def hz_to_mel(hz):
    """
    Convert frequency in Hz to Mel scale
    
    Math: Mel(f) = 2595 × log10(1 + f/700)
    """
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    """
    Convert Mel scale back to Hz
    
    Math: f = 700 × (10^(Mel/2595) - 1)
    """
    return 700 * (10 ** (mel / 2595) - 1)

def create_mel_filterbank(num_filters, nfft, sample_rate, low_freq=0, high_freq=None):
    """
    Manually create Mel filter bank (triangular filters)
    
    Math: Creates overlapping triangular filters in Mel scale
    """
    if high_freq is None:
        high_freq = sample_rate / 2
    
    # Convert Hz to Mel
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    
    # Create num_filters+2 equally spaced points in Mel scale
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    
    # Convert back to Hz
    hz_points = mel_to_hz(mel_points)
    
    # Convert Hz to FFT bin numbers
    bin_points = np.floor((nfft + 1) * hz_points / sample_rate).astype(int)
    
    # Create filter bank matrix
    num_fft_bins = nfft // 2 + 1
    filterbank = np.zeros((num_filters, num_fft_bins))
    
    # Create triangular filters
    for m in range(1, num_filters + 1):
        left = bin_points[m - 1]    # Left edge
        center = bin_points[m]       # Peak
        right = bin_points[m + 1]    # Right edge
        
        # Rising slope
        for k in range(left, center):
            if center != left:
                filterbank[m - 1, k] = (k - left) / (center - left)
        
        # Falling slope
        for k in range(center, right):
            if right != center:
                filterbank[m - 1, k] = (right - k) / (right - center)
    
    return filterbank

# Create Mel filter bank
print(f"\n[3.4.1] Creating Mel Filter Bank...")
mel_filterbank = create_mel_filterbank(
    num_filters=NUM_MEL_FILTERS,
    nfft=FRAME_SIZE,
    sample_rate=sample_rate,
    low_freq=0,
    high_freq=sample_rate / 2
)

print(f"        Filter Bank Shape: {mel_filterbank.shape}")
print(f"        Number of Filters: {NUM_MEL_FILTERS}")
print(f"        Frequency Range: 0 Hz to {sample_rate/2:.1f} Hz (Nyquist)")

# Apply Mel filter bank to power spectrum
print(f"\n[3.4.2] Applying Mel Filters to Power Spectrum...")
print(f"        Power Spectrum Shape: {power_spectrum.shape}")
print(f"        Operation: Matrix multiplication (frames × filters)")

# Filter bank energies = Power Spectrum × Mel Filter Bank^T
filter_bank_energies = np.dot(power_spectrum, mel_filterbank.T)

print(f"        Filter Bank Energies Shape: {filter_bank_energies.shape}")
print(f"        Result: {filter_bank_energies.shape[0]} frames × {NUM_MEL_FILTERS} Mel bands")
print(f"        ✓ Converted from linear to Mel frequency scale")

# =============================================================================
# STEP 3.5: LOG MEL SPECTRUM
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3.5: LOG MEL SPECTRUM")
print("=" * 80)

print(f"\n[MATH CONCEPT - LOGARITHMIC COMPRESSION]:")
print(f"  • Formula: log(E + ε) where E is energy, ε prevents log(0)")
print(f"  • Purpose: Compress dynamic range (human hearing is logarithmic)")
print(f"  • Effect: Makes quiet sounds more prominent, loud sounds less dominant")
print(f"  • Epsilon (ε): Small value to avoid log(0) = -infinity")

print(f"\n[3.5.1] Applying Logarithm to Filter Bank Energies...")

# Add small epsilon to prevent log(0)
epsilon = 1e-10
log_mel_spectrum = np.log(filter_bank_energies + epsilon)

print(f"        Formula used: log(energy + {epsilon})")
print(f"        Before log - Range: [{filter_bank_energies.min():.6f}, {filter_bank_energies.max():.6f}]")
print(f"        After log  - Range: [{log_mel_spectrum.min():.6f}, {log_mel_spectrum.max():.6f}]")
print(f"        ✓ Logarithmic compression applied")

# =============================================================================
# STEP 3.6: DCT (DISCRETE COSINE TRANSFORM) - MFCC EXTRACTION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3.6: DCT (DISCRETE COSINE TRANSFORM) - MFCC")
print("=" * 80)

print(f"\n[MATH CONCEPT - DCT]:")
print(f"  • Formula: MFCC[n] = Σ log_Mel[k] × cos(π×n×(k+0.5)/K)")
print(f"  • n: MFCC coefficient index (0 to 12 for 13 coefficients)")
print(f"  • k: Mel filter bank index (0 to {NUM_MEL_FILTERS-1})")
print(f"  • K: Total number of Mel filters ({NUM_MEL_FILTERS})")
print(f"  • Purpose: De-correlate Mel coefficients, compress to fewer features")

def manual_dct(signal, num_coeffs):
    """
    Manually implement DCT Type-II (used in MFCC)
    
    Math: DCT[n] = Σ signal[k] × cos(π × n × (k + 0.5) / K)
          for n = 0, 1, ..., num_coeffs-1
          for k = 0, 1, ..., K-1
    """
    num_frames, num_filters = signal.shape
    dct_coeffs = np.zeros((num_frames, num_coeffs))
    
    for n in range(num_coeffs):
        for k in range(num_filters):
            # DCT-II formula
            dct_basis = np.cos(np.pi * n * (k + 0.5) / num_filters)
            dct_coeffs[:, n] += signal[:, k] * dct_basis
    
    return dct_coeffs

print(f"\n[3.6.1] Applying DCT to Log Mel Spectrum...")
print(f"        Input: {log_mel_spectrum.shape[0]} frames × {NUM_MEL_FILTERS} Mel bands")
print(f"        Extracting first {NUM_MFCC_COEFFS} MFCC coefficients per frame")

# Apply DCT manually
mfcc_features = manual_dct(log_mel_spectrum, NUM_MFCC_COEFFS)

print(f"\n[3.6.2] MFCC Features Extracted")
print(f"        Output Shape: {mfcc_features.shape}")
print(f"        {mfcc_features.shape[0]} frames × {NUM_MFCC_COEFFS} MFCC coefficients")
print(f"        Frame 0 MFCCs (first 5): {mfcc_features[0, :5]}")
print(f"        ✓ Full MFCC pipeline complete!")

print(f"\n[3.6.3] Feature Compression Summary")
print(f"        Original: {power_spectrum.shape[1]} frequency bins")
print(f"        After Mel Filtering: {NUM_MEL_FILTERS} Mel bands")
print(f"        After DCT: {NUM_MFCC_COEFFS} MFCC coefficients")
print(f"        Compression Ratio: {power_spectrum.shape[1] / NUM_MFCC_COEFFS:.1f}:1")

# =============================================================================
# STEP 4: ACOUSTIC & LANGUAGE MODELS (CONCEPTUAL EXPLANATION)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: ACOUSTIC MODEL & LANGUAGE MODEL (CONCEPTUAL)")
print("=" * 80)

print(f"\n[4.1] ACOUSTIC MODEL:")
print(f"      • Input: Features from Step 3 (Power Spectrum / MFCC)")
print(f"      • Architecture: Deep Neural Network (DNN, RNN, or Transformer)")
print(f"      • Purpose: Maps audio features → phonemes/characters")
print(f"      • Training: Learns patterns from labeled speech data")
print(f"      • Output: Probability distribution over possible phonemes/words")
print(f"")
print(f"      Example: Power spectrum → [0.7 'h', 0.2 'ae', 0.1 'k']")

print(f"\n[4.2] LANGUAGE MODEL:")
print(f"      • Input: Phoneme/word probabilities from Acoustic Model")
print(f"      • Architecture: N-gram model or Neural Language Model (LSTM/GPT)")
print(f"      • Purpose: Apply linguistic context and grammar rules")
print(f"      • Training: Learns word sequences from large text corpus")
print(f"      • Output: Most likely sentence given audio + language context")
print(f"")
print(f"      Example: ['recognize', 'wreck a nice'] → 'recognize' (higher probability)")

print(f"\n[4.3] DECODING (COMBINING BOTH MODELS):")
print(f"      • Formula: W* = argmax P(W|X) = argmax P(X|W) * P(W)")
print(f"               W*    = best word sequence")
print(f"               P(X|W) = Acoustic Model probability")
print(f"               P(W)   = Language Model probability")
print(f"")
print(f"      • Algorithm: Beam Search or Viterbi Decoding")
print(f"      • Result: Final transcribed text")

# =============================================================================
# STEP 5: SAVE PROCESSED AUDIO (OPTIONAL)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: SAVE PROCESSED AUDIO")
print("=" * 80)

# Convert back to int16 for saving
print(f"\n[5.1] Preparing Audio for WAV Export")

# Normalize to prevent clipping
print(f"      Step 1: Normalize to prevent clipping")
max_val = np.max(np.abs(audio_no_silence))
print(f"      Max absolute value: {max_val:.6f}")

if max_val > 0:
    # Scale to 95% to leave headroom
    audio_normalized = audio_no_silence / max_val * 0.95
    print(f"      Normalized to 95% of full scale")
else:
    audio_normalized = audio_no_silence
    print(f"      ⚠️  WARNING: Output audio is silent")

print(f"      Normalized range: [{audio_normalized.min():.6f}, {audio_normalized.max():.6f}]")

# Convert to int16 (WAV format requirement)
print(f"\n      Step 2: Convert float32 → int16")
print(f"      Formula: int16_value = float_value × 32767")
audio_output = np.int16(audio_normalized * 32767)

print(f"      Data type: {audio_output.dtype}")
print(f"      Range: [{audio_output.min()}, {audio_output.max()}]")
print(f"      ✅ Type conversion successful (float32 → int16)")

# Save to file
wavfile.write(OUTPUT_FILE, sample_rate, audio_output)

print(f"\n[5.2] Processed Audio Saved")
print(f"      File: {OUTPUT_FILE}")
print(f"      Sampling Rate: {sample_rate} Hz")
print(f"      Duration: {len(audio_output) / sample_rate:.2f} seconds")
print(f"      Samples: {len(audio_output):,}")
print(f"      Format: 16-bit PCM WAV")
print(f"      ✅ File saved successfully")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("PIPELINE SUMMARY")
print("=" * 80)

print(f"""
Step 1: ADC Simulation & Preprocessing
        ✓ Loaded: {original_sample_rate} Hz → Downsampled to {sample_rate} Hz
        ✓ Stereo ({audio_data.ndim} channels) → Mono conversion
        ✓ Decimation: Every {DECIMATION_FACTOR}rd sample (44.1kHz → ~14.7kHz)
        ✓ Normalized to float [-1, 1]

Step 2: Audio Cleaning
        ✓ Noise Reduction: Moving Average Filter (window={MOVING_AVG_WINDOW})
        ✓ Silence Removal: Dynamic threshold ({SILENCE_THRESHOLD_RATIO*100}% of mean energy)
        ✓ Removed {removed_count} frames ({removed_count/(len(audio_denoised)//FRAME_SIZE)*100:.1f}%)

Step 3: Feature Extraction (FULL MFCC PIPELINE)
        ✓ Pre-emphasis Filter (α={PRE_EMPHASIS_COEF})
        ✓ Framing: {len(frames)} frames ({FRAME_SIZE/sample_rate*1000:.1f}ms each)
        ✓ Windowing: Hamming window applied
        ✓ FFT: {power_spectrum.shape[1]} frequency bins per frame
        ✓ Mel Filter Banks: {NUM_MEL_FILTERS} triangular filters
        ✓ Log Mel Spectrum: Logarithmic compression applied
        ✓ DCT: Extracted {NUM_MFCC_COEFFS} MFCC coefficients per frame
        ✓ Final MFCC Output: {mfcc_features.shape}

Step 4: Models (Conceptual)
        ✓ Acoustic Model: MFCC Features → Phonemes/Words
        ✓ Language Model: Context → Final Transcription

Step 5: Output
        ✓ Audio saved to: {OUTPUT_FILE}
        ✓ MFCC features ready for acoustic model input
        ✓ Shape: {mfcc_features.shape[0]} frames × {NUM_MFCC_COEFFS} coefficients

All steps implemented MANUALLY from scratch using only NumPy & SciPy!
No librosa, noisereduce, or torchaudio used.
Full MFCC extraction pipeline complete!
""")

print("=" * 80)
print("PIPELINE COMPLETE!")
print("=" * 80)

print("\n" + "=" * 80)
print("IMPORTANT NOTE - SCOPE OF THIS IMPLEMENTATION")
print("=" * 80)
print("""
This script demonstrates SIGNAL PREPROCESSING and FEATURE EXTRACTION.

✓ What This Script Does:
  - Converts Stereo → Mono
  - Downsamples 44.1kHz → ~14.7kHz (for ASR efficiency)
  - Removes noise (Moving Average Filter)
  - Removes silence (Short-Time Energy with dynamic threshold)
  - Extracts features (Pre-emphasis, Framing, Windowing, FFT)

✗ What This Script Does NOT Do (Requires Deep Learning):
  - Text Transcription (needs trained Acoustic Model)
  - Word Recognition (needs trained Neural Network)
  - Language Understanding (needs Language Model)

🎯 To Get Actual Text from Audio, You Need:

  1. ACOUSTIC MODEL (Deep Neural Network)
     - Architecture: CNN, RNN, LSTM, Transformer, or Wav2Vec
     - Training: Thousands of hours of labeled speech data
     - Hardware: High-end GPUs (weeks/months of training)
     - Output: Phoneme or character probabilities
  
  2. LANGUAGE MODEL
     - N-gram model or Neural LM (GPT, BERT-based)
     - Trained on large text corpora
     - Corrects transcriptions using context and grammar
  
  3. DECODER
     - Beam Search or CTC (Connectionist Temporal Classification)
     - Combines acoustic + language model
     - Produces final text

📚 For Production ASR Systems, Use:
  - OpenAI Whisper (state-of-the-art, multilingual)
  - Meta's Wav2Vec 2.0 (self-supervised learning)
  - Google's Speech-to-Text API
  - Mozilla DeepSpeech
  - Kaldi, ESPnet (research frameworks)

This educational script shows the MATHEMATICAL FOUNDATIONS that prepare
audio data BEFORE it enters those complex neural network systems.
""")

print("=" * 80)


# ============================================================================
# STEP 6: VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: VISUALIZING MFCC FEATURES")
print("=" * 80)

# Transpose the MFCC matrix for proper visualization
# Original shape: (num_frames, num_coefficients) = (60960, 13)
# Transposed shape: (num_coefficients, num_frames) = (13, 60960)
mfcc_transposed = mfcc_features.T

print(f"\n📊 MFCC Matrix for Visualization:")
print(f"  - Original shape: {mfcc_features.shape} (frames × coefficients)")
print(f"  - Transposed shape: {mfcc_transposed.shape} (coefficients × frames)")
print(f"  - Value range: [{mfcc_transposed.min():.3f}, {mfcc_transposed.max():.3f}]")

# Create the MFCC spectrogram visualization
plt.figure(figsize=(12, 6))
img = plt.imshow(mfcc_transposed, aspect='auto', origin='lower', cmap='viridis')

# Add colorbar
plt.colorbar(img, label='MFCC Magnitude')

# Add labels and title
plt.xlabel("Time Frames", fontsize=12)
plt.ylabel("MFCC Coefficients", fontsize=12)
plt.title("MFCC Spectrogram", fontsize=14, fontweight='bold')

# Save the figure
output_image_path = "mfcc_output.png"
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
print(f"\n💾 MFCC spectrogram saved to: {output_image_path}")

# Display the plot
print("\n🎨 Displaying MFCC visualization...")
plt.show()

print("=" * 80)
