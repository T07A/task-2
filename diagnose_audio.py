"""
Audio File Diagnostic Tool
==========================
Analyzes audio files to detect common issues with sampling rate, 
channels, duration, and amplitude that can cause playback problems.
"""

import numpy as np
from scipy.io import wavfile

print("=" * 80)
print("AUDIO FILE HEALTH REPORT")
print("=" * 80)

# Files to analyze
INPUT_FILE = "input.wav"
OUTPUT_FILE = "processed_manual.wav"

def analyze_audio_file(filename):
    """
    Analyze an audio file and return diagnostic information
    """
    try:
        # Read the file
        sample_rate, data = wavfile.read(filename)
        
        # Determine if Mono or Stereo
        if len(data.shape) == 1:
            channels = "Mono"
            num_channels = 1
            data_mono = data
        else:
            channels = f"Stereo ({data.shape[1]} channels)"
            num_channels = data.shape[1]
            # Use first channel for amplitude analysis
            data_mono = data[:, 0]
        
        # Calculate duration
        if len(data.shape) == 1:
            num_samples = len(data)
        else:
            num_samples = data.shape[0]
        
        duration = num_samples / sample_rate
        
        # Amplitude analysis
        min_amplitude = np.min(data)
        max_amplitude = np.max(data)
        abs_max = np.max(np.abs(data))
        
        # Normalized amplitude (if it's int16)
        if data.dtype == np.int16:
            normalized_max = abs_max / 32768.0
        else:
            normalized_max = abs_max
        
        return {
            'sample_rate': sample_rate,
            'channels': channels,
            'num_channels': num_channels,
            'duration': duration,
            'num_samples': num_samples,
            'min_amplitude': min_amplitude,
            'max_amplitude': max_amplitude,
            'abs_max': abs_max,
            'normalized_max': normalized_max,
            'data_type': data.dtype,
            'shape': data.shape
        }
    
    except FileNotFoundError:
        return None
    except Exception as e:
        return {'error': str(e)}

# Analyze INPUT file
print(f"\n{'='*80}")
print(f"FILE 1: {INPUT_FILE}")
print(f"{'='*80}")

input_info = analyze_audio_file(INPUT_FILE)

if input_info is None:
    print(f"❌ ERROR: File '{INPUT_FILE}' not found!")
elif 'error' in input_info:
    print(f"❌ ERROR: {input_info['error']}")
else:
    print(f"\n📊 BASIC INFO:")
    print(f"   Sampling Rate:    {input_info['sample_rate']:,} Hz")
    print(f"   Channels:         {input_info['channels']}")
    print(f"   Duration:         {input_info['duration']:.2f} seconds")
    print(f"   Total Samples:    {input_info['num_samples']:,}")
    print(f"   Data Type:        {input_info['data_type']}")
    print(f"   Array Shape:      {input_info['shape']}")
    
    print(f"\n📈 AMPLITUDE INFO:")
    print(f"   Min Value:        {input_info['min_amplitude']}")
    print(f"   Max Value:        {input_info['max_amplitude']}")
    print(f"   Max Absolute:     {input_info['abs_max']}")
    print(f"   Normalized Max:   {input_info['normalized_max']:.4f} (scale 0-1)")

# Analyze OUTPUT file
print(f"\n{'='*80}")
print(f"FILE 2: {OUTPUT_FILE}")
print(f"{'='*80}")

output_info = analyze_audio_file(OUTPUT_FILE)

if output_info is None:
    print(f"❌ ERROR: File '{OUTPUT_FILE}' not found!")
    print(f"   (This is normal if you haven't run the processing script yet)")
elif 'error' in output_info:
    print(f"❌ ERROR: {output_info['error']}")
else:
    print(f"\n📊 BASIC INFO:")
    print(f"   Sampling Rate:    {output_info['sample_rate']:,} Hz")
    print(f"   Channels:         {output_info['channels']}")
    print(f"   Duration:         {output_info['duration']:.2f} seconds")
    print(f"   Total Samples:    {output_info['num_samples']:,}")
    print(f"   Data Type:        {output_info['data_type']}")
    print(f"   Array Shape:      {output_info['shape']}")
    
    print(f"\n📈 AMPLITUDE INFO:")
    print(f"   Min Value:        {output_info['min_amplitude']}")
    print(f"   Max Value:        {output_info['max_amplitude']}")
    print(f"   Max Absolute:     {output_info['abs_max']}")
    print(f"   Normalized Max:   {output_info['normalized_max']:.4f} (scale 0-1)")

# COMPARISON & DIAGNOSIS
if input_info and output_info and 'error' not in input_info and 'error' not in output_info:
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPARISON")
    print(f"{'='*80}")
    
    # Check 1: Sampling Rate Mismatch
    print(f"\n🔍 CHECK 1: SAMPLING RATE MISMATCH (Chipmunk Effect)")
    if input_info['sample_rate'] != output_info['sample_rate']:
        print(f"   ⚠️  WARNING: Sampling rates DO NOT MATCH!")
        print(f"       Input:  {input_info['sample_rate']:,} Hz")
        print(f"       Output: {output_info['sample_rate']:,} Hz")
        print(f"   ")
        print(f"   🐿️ This WILL cause speed distortion (chipmunk/slow effect)!")
        print(f"   ")
        ratio = output_info['sample_rate'] / input_info['sample_rate']
        if ratio < 1:
            print(f"       Playback will be {1/ratio:.2f}x FASTER than original")
        else:
            print(f"       Playback will be {ratio:.2f}x SLOWER than original")
    else:
        print(f"   ✅ PASS: Sampling rates match ({input_info['sample_rate']:,} Hz)")
    
    # Check 2: Channel Mismatch
    print(f"\n🔍 CHECK 2: CHANNEL MISMATCH")
    if input_info['num_channels'] != output_info['num_channels']:
        print(f"   ⚠️  WARNING: Channel counts differ!")
        print(f"       Input:  {input_info['channels']}")
        print(f"       Output: {output_info['channels']}")
    else:
        print(f"   ✅ PASS: Both files have {output_info['channels']}")
    
    # Check 3: Duration Change (Silence Removal)
    print(f"\n🔍 CHECK 3: DURATION CHANGE (Silence Removal Effect)")
    duration_diff = input_info['duration'] - output_info['duration']
    duration_percent = (duration_diff / input_info['duration']) * 100
    
    print(f"   Input Duration:   {input_info['duration']:.2f} seconds")
    print(f"   Output Duration:  {output_info['duration']:.2f} seconds")
    print(f"   Time Removed:     {duration_diff:.2f} seconds ({duration_percent:.1f}%)")
    
    if duration_percent > 50:
        print(f"   ⚠️  WARNING: More than 50% of audio removed!")
        print(f"       Silence threshold may be too aggressive")
        print(f"       This can cause choppy/cut-off speech")
    elif duration_percent > 20:
        print(f"   ⚠️  CAUTION: {duration_percent:.1f}% removed - verify quality")
    else:
        print(f"   ✅ PASS: Normal silence removal range")
    
    # Check 4: Amplitude/Volume Check
    print(f"\n🔍 CHECK 4: AMPLITUDE/VOLUME ANALYSIS")
    print(f"   Input Max (normalized):  {input_info['normalized_max']:.4f}")
    print(f"   Output Max (normalized): {output_info['normalized_max']:.4f}")
    
    if output_info['normalized_max'] < 0.1:
        print(f"   ⚠️  WARNING: Output volume very low ({output_info['normalized_max']:.4f})")
        print(f"       May be too quiet for proper silence detection")
    elif output_info['normalized_max'] > 0.95:
        print(f"   ✅ PASS: Good output volume (normalized to ~{output_info['normalized_max']:.2f})")
    else:
        print(f"   ✅ PASS: Acceptable output volume")

print(f"\n{'='*80}")
print(f"DIAGNOSIS COMPLETE")
print(f"{'='*80}")
print(f"""
💡 INTERPRETATION GUIDE:
   
   If you hear CHIPMUNK/FAST audio:
   → Check sampling rate mismatch (Input Hz ≠ Output Hz)
   → The script may be forcing 16kHz when input is 44.1kHz
   
   If you hear CHOPPY/CUT-OFF speech:
   → Check duration reduction percentage
   → Lower the silence removal threshold
   → Use adaptive threshold based on max energy
   
   If audio is SILENT or TOO QUIET:
   → Check amplitude values (should be close to 1.0 when normalized)
   → May need to adjust normalization
""")
