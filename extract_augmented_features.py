"""
Extract Features from Augmented Audio Files
===========================================

This script extracts the same 36 features from augmented audio files
and saves them to a CSV file that can be used with your existing training code.

Usage:
    python extract_augmented_features.py
"""

import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIG ----------
INPUT_DIR = "output/augmented_lung"
OUTPUT_CSV = "augmented_lung_features.csv"

# ---------- FEATURE EXTRACTION (YOUR EXACT METHOD) ----------
def extract_features(file_path):
    """Extract 36 features from audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5)
        
        # MFCCs (13 coefficients) - Mean & Std
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]
        
        # Combine all 36 features
        features = []
        features.extend(mfccs_mean)      # 13
        features.extend(mfccs_std)       # 13
        features.append(np.mean(spec_cent))
        features.append(np.std(spec_cent))
        features.append(np.mean(spec_bw))
        features.append(np.std(spec_bw))
        features.append(np.mean(spec_rolloff))
        features.append(np.std(spec_rolloff))
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        features.append(np.mean(rms))
        features.append(np.std(rms))    # 10
        
        return features
        
    except Exception as e:
        return None

# ---------- MAIN ----------
def main():
    print("\n" + "="*60)
    print("🔬 EXTRACTING FEATURES FROM AUGMENTED AUDIO")
    print("="*60 + "\n")
    
    # Find all augmented audio files
    audio_files = []
    if os.path.exists(INPUT_DIR):
        for f in os.listdir(INPUT_DIR):
            if f.endswith('.wav'):
                audio_files.append(os.path.join(INPUT_DIR, f))
    else:
        print(f"❌ Directory not found: {INPUT_DIR}")
        return
    
    print(f"📁 Found {len(audio_files)} augmented audio files")
    
    if len(audio_files) == 0:
        print("❌ No .wav files found. Did the augmentation run successfully?")
        return
    
    # Extract features
    print("\n🎵 Extracting features...")
    all_features = []
    all_filenames = []
    
    # Get feature names from first file
    first_features = extract_features(audio_files[0])
    if first_features is None:
        print("❌ Failed to extract features from first file")
        return
    
    feature_names = (
        [f'MFCC_{i+1}' for i in range(13)] +
        [f'MFCC_std_{i+1}' for i in range(13)] +
        ['spec_cent_mean', 'spec_cent_std', 'spec_bw_mean', 'spec_bw_std',
         'spec_rolloff_mean', 'spec_rolloff_std', 'zcr_mean', 'zcr_std',
         'rms_mean', 'rms_std']
    )
    
    # Process all files
    for file_path in tqdm(audio_files, desc="Extracting"):
        features = extract_features(file_path)
        if features is not None:
            all_features.append(features)
            all_filenames.append(os.path.basename(file_path))
    
    if len(all_features) == 0:
        print("❌ No features extracted")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_features, columns=feature_names)
    df.insert(0, 'Filename', all_filenames)
    
    # Add Patient_ID from the original filename
    # Your filenames like: 103_2b2_Ar_s__1_aug_1.wav → Patient_ID = 103
    df['Patient_ID'] = df['Filename'].str.extract(r'^(\d+)').astype(int)
    
    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Saved {len(df)} samples to {OUTPUT_CSV}")
    print(f"   Shape: {df.shape}")
    print("\n📊 Preview:")
    print(df[['Filename', 'Patient_ID'] + feature_names[:3]].head())
    
    return df

if __name__ == "__main__":
    main()