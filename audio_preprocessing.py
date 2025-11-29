"""
Audio Preprocessing Pipeline
A simple academic script demonstrating audio preprocessing steps
"""

import librosa
import noisereduce as nr
import numpy as np
from scipy.io import wavfile

# File paths
INPUT_FILE = "casadt_000127_access.wav"
OUTPUT_FILE = "processed_audio.wav"

print("=" * 60)
print("Audio Preprocessing Pipeline")
print("=" * 60)

# Step 1: Analog-to-Digital Simulation (Loading)
# We use 16kHz (16000 Hz) sampling rate because:
# - It's the standard for speech processing and recognition
# - Captures frequencies up to 8kHz (human speech is mostly 300Hz-3400Hz)
# - Balances quality with computational efficiency
print("\n[Step 1] Loading audio file...")
audio_data, sample_rate = librosa.load(INPUT_FILE, sr=16000)
print(f"Loaded: {INPUT_FILE}")
print(f"Sampling Rate: {sample_rate} Hz")
print(f"Original Duration: {len(audio_data) / sample_rate:.2f} seconds")
print(f"Audio Shape: {audio_data.shape}")

# Step 2: Noise Reduction
# noisereduce library uses spectral gating to reduce background noise
# It analyzes the frequency spectrum and filters out consistent noise patterns
print("\n[Step 2] Applying noise reduction...")
reduced_noise_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)

# Check for NaN values and clean them
if np.isnan(reduced_noise_audio).any():
    print("WARNING: NaN values detected after noise reduction. Cleaning...")
    reduced_noise_audio = np.nan_to_num(reduced_noise_audio, nan=0.0)

print("Noise reduction completed using spectral gating")

# Step 3: Silence Removal
# librosa.effects.trim removes silent portions from start and end
# top_db=20 means audio below 20 decibels from peak is considered silence
# Lower values (e.g., 30) are more aggressive in removing quiet sections
print("\n[Step 3] Trimming silence from beginning and end...")
trimmed_audio, _ = librosa.effects.trim(reduced_noise_audio, top_db=30)

# Safety check: ensure we have audio data left
if len(trimmed_audio) < sample_rate * 0.1:  # Less than 0.1 seconds
    print("WARNING: Trimming removed too much audio. Using non-trimmed version.")
    trimmed_audio = reduced_noise_audio

print(f"Processed Duration: {len(trimmed_audio) / sample_rate:.2f} seconds")
print(f"Time Removed: {(len(reduced_noise_audio) - len(trimmed_audio)) / sample_rate:.2f} seconds")

# Step 4: Save Output
# scipy.io.wavfile is used to write the processed audio back to disk
# We convert to 16-bit PCM format (standard for WAV files)
print("\n[Step 4] Saving processed audio...")

# Clean any remaining NaN or inf values
trimmed_audio = np.nan_to_num(trimmed_audio, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize to prevent clipping
max_val = np.max(np.abs(trimmed_audio))
if max_val > 0:
    trimmed_audio = trimmed_audio / max_val * 0.95

# Normalize audio to int16 range (-32768 to 32767)
trimmed_audio_int16 = np.int16(trimmed_audio * 32767)
wavfile.write(OUTPUT_FILE, sample_rate, trimmed_audio_int16)
print(f"Saved: {OUTPUT_FILE}")

print("\n" + "=" * 60)
print("Preprocessing Complete!")
print("=" * 60)
print(f"\nSummary:")
print(f"  Original Duration: {len(audio_data) / sample_rate:.2f}s")
print(f"  Final Duration: {len(trimmed_audio) / sample_rate:.2f}s")
print(f"  Reduction: {((len(audio_data) - len(trimmed_audio)) / len(audio_data) * 100):.1f}%")
