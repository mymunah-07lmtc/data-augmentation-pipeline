"""
Data Augmentation Pipeline for Sahel Sound Triage Platform
===========================================================

This script applies realistic audio augmentations to the ICBHI 2017
respiratory sound dataset and CirCor heart sound dataset to simulate
real-world conditions in rural Mali clinics.

Augmentations applied:
- Background noise (synthetic: white, brown, pink noise)
- Speed variation (0.8x - 1.2x)
- Volume variation (±10dB)
- Filtering (simulate low-quality microphones)
- Time shift (random)
- Pitch shift (±2 semitones)

Outputs:
- Augmented audio files (saved to output/augmented_lung/ and output/augmented_heart/)
- CSV log mapping original → augmented files
- Dataset statistics
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# For augmentation
try:
    import audiomentations as A
    HAS_AUDIOMENTATIONS = True
except ImportError:
    HAS_AUDIOMENTATIONS = False
    print("⚠️  audiomentations not installed.")
    print("   Run: pip install audiomentations")

# ---------- CONFIGURATION ----------
CONFIG = {
    "input_lung_dir": "data/ICBHI_2017",  # Path to ICBHI dataset
    "input_heart_dir": "data/CirCor",     # Path to CirCor dataset
    "output_base_dir": "output",
    "num_augmentations": 5,                # Number of augmentations per file
    "target_sr": 16000,                    # Target sample rate
    "duration": 5.0,                       # Target duration in seconds
    "seed": 42,                            # For reproducibility
}

# ---------- AUGMENTATION PIPELINE ----------
class AudioAugmentationPipeline:
    """Pipeline for applying realistic augmentations to medical audio."""
    
    def __init__(self, sample_rate=16000, duration=5.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Define the augmentation pipeline."""
        self.pipeline = A.Compose([
            # 1. Speed variation (breathing rate changes)
            A.TimeStretch(
                min_rate=0.8,
                max_rate=1.2,
                p=0.6
            ),
            
            # 2. Volume variation (±10dB)   <-- FIXED parameter names
            A.Gain(
                min_gain_db=-10,      # was min_gain_in_db
                max_gain_db=10,       # was max_gain_in_db
                p=0.7
            ),
            
            # 3. Filtering (low-quality microphone)
            A.HighPassFilter(
                min_cutoff_freq=50,
                max_cutoff_freq=150,
                p=0.5
            ),
            
            # 4. Time shift (alignment variation)
            A.Shift(
                min_shift=-0.2,
                max_shift=0.2,
                p=0.5
            ),
            
            # 5. Pitch shift (vocal variation)
            A.PitchShift(
                min_semitones=-2,
                max_semitones=2,
                p=0.4
            ),
            
            # 6. Add Gaussian noise (simulates electronic interference)
            A.AddGaussianNoise(
                min_amplitude=0.001,
                max_amplitude=0.015,
                p=0.6
            ),
        ], p=1.0)
    
    def generate_noise(self, duration):
        """Generate synthetic noise for augmentation."""
        # Generate random noise types
        noise_type = random.choice(['white', 'brown', 'pink'])
        if noise_type == 'white':
            noise = np.random.normal(0, 0.1, int(duration * self.sample_rate))
        elif noise_type == 'brown':
            # Brownian noise (1/f^2)
            white = np.random.normal(0, 0.1, int(duration * self.sample_rate))
            noise = np.cumsum(white) / np.sqrt(len(white))
        else:  # pink noise (1/f)
            white = np.random.normal(0, 0.1, int(duration * self.sample_rate))
            # Simple pink noise approximation
            noise = np.fft.irfft(
                np.fft.rfft(white) / np.sqrt(np.arange(1, len(np.fft.rfft(white)) + 1))
            )
            noise = noise[:len(white)]
        return noise
    
    def apply(self, audio):
        """Apply augmentation pipeline to audio."""
        # Ensure correct duration
        if len(audio) > self.duration * self.sample_rate:
            audio = audio[:int(self.duration * self.sample_rate)]
        else:
            padding = int(self.duration * self.sample_rate) - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Generate background noise
        noise = self.generate_noise(self.duration)
        
        # Apply pipeline
        augmented = self.pipeline(
            samples=audio.astype(np.float32),
            sample_rate=self.sample_rate
        )
        
        # Add background noise (custom)
        noise_amplitude = 0.1 * random.uniform(0.5, 1.5)
        augmented = augmented + noise_amplitude * noise
        
        # Clip to prevent distortion
        augmented = np.clip(augmented, -1.0, 1.0)
        
        return augmented.astype(np.float32)


# ---------- MAIN PROCESSING ----------
def process_dataset(input_dir, output_dir, pipeline, num_augmentations=5):
    """
    Process all audio files in a directory with augmentations.
    
    Args:
        input_dir: Path to input directory containing audio files
        output_dir: Path to output directory for augmented files
        pipeline: AudioAugmentationPipeline instance
        num_augmentations: Number of augmentations per file
    
    Returns:
        DataFrame with mapping original → augmented files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files
    audio_files = []
    for ext in ['.wav', '.flac', '.mp3']:
        audio_files.extend(glob.glob(os.path.join(input_dir, f'**/*{ext}'), recursive=True))
    
    if not audio_files:
        print(f"⚠️  No audio files found in {input_dir}")
        return pd.DataFrame()
    
    print(f"📁 Found {len(audio_files)} audio files in {input_dir}")
    
    # Process each file
    records = []
    for file_path in tqdm(audio_files, desc="Processing files"):
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=CONFIG['target_sr'])
            if len(audio) == 0:
                continue
            
            # Generate augmentations
            for i in range(num_augmentations):
                augmented = pipeline.apply(audio)
                
                # Save augmented file
                base_name = os.path.basename(file_path).rsplit('.', 1)[0]
                aug_name = f"{base_name}_aug_{i+1}.wav"
                aug_path = os.path.join(output_dir, aug_name)
                sf.write(aug_path, augmented, CONFIG['target_sr'])
                
                # Record mapping
                records.append({
                    'original_file': file_path,
                    'augmented_file': aug_path,
                    'augmentation_id': i + 1,
                    'original_label': os.path.basename(os.path.dirname(file_path)),
                    'duration': len(augmented) / CONFIG['target_sr']
                })
        
        except Exception as e:
            print(f"⚠️  Error processing {file_path}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(records)
    return df


# ---------- DATASET STATISTICS ----------
def generate_statistics(df, output_dir):
    """Generate and save dataset statistics."""
    if df.empty:
        return
    
    stats = {
        'total_files': len(df),
        'unique_originals': df['original_file'].nunique(),
        'augmentations_per_file': df['augmentation_id'].max(),
        'total_duration_seconds': df['duration'].sum(),
        'total_duration_hours': df['duration'].sum() / 3600,
        'label_counts': df['original_label'].value_counts().to_dict()
    }
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(os.path.join(output_dir, 'dataset_statistics.csv'), index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    print(f"Total augmented files: {stats['total_files']}")
    print(f"Original files: {stats['unique_originals']}")
    print(f"Augmentations per original: {stats['augmentations_per_file']}")
    print(f"Total duration: {stats['total_duration_hours']:.2f} hours")
    print("\nLabels:")
    for label, count in stats['label_counts'].items():
        print(f"  - {label}: {count}")
    print("="*50)
    
    return stats


# ---------- MAIN EXECUTION ----------
def main():
    """Main entry point."""
    print("\n" + "="*50)
    print("🎵 DATA AUGMENTATION PIPELINE")
    print("   Sahel Sound Triage Platform")
    print("="*50 + "\n")
    
    # Setup
    random.seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Create output directories
    lung_output = os.path.join(CONFIG['output_base_dir'], 'augmented_lung')
    heart_output = os.path.join(CONFIG['output_base_dir'], 'augmented_heart')
    os.makedirs(lung_output, exist_ok=True)
    os.makedirs(heart_output, exist_ok=True)
    
    # Initialize pipeline
    pipeline = AudioAugmentationPipeline(
        sample_rate=CONFIG['target_sr'],
        duration=CONFIG['duration']
    )
    
    # Process lung dataset
    print("\n🫁 Processing lung sounds (ICBHI 2017)...")
    lung_df = process_dataset(
        CONFIG['input_lung_dir'],
        lung_output,
        pipeline,
        CONFIG['num_augmentations']
    )
    
    # Process heart dataset
    print("\n❤️  Processing heart sounds (CirCor)...")
    heart_df = process_dataset(
        CONFIG['input_heart_dir'],
        heart_output,
        pipeline,
        CONFIG['num_augmentations']
    )
    
    # Combine and save
    all_df = pd.concat([lung_df, heart_df], ignore_index=True)
    if not all_df.empty:
        log_path = os.path.join(CONFIG['output_base_dir'], 'augmentation_log.csv')
        all_df.to_csv(log_path, index=False)
        print(f"\n✅ Augmentation log saved to: {log_path}")
        
        # Generate statistics
        generate_statistics(all_df, CONFIG['output_base_dir'])
    
    # Summary
    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETE")
    print(f"   Lung files augmented: {len(lung_df)}")
    print(f"   Heart files augmented: {len(heart_df)}")
    print(f"   Total augmented files: {len(all_df)}")
    print("="*50 + "\n")
    
    # Next steps
    print("📌 Next steps:")
    print("1. Train your model on the augmented dataset")
    print("2. Compare performance with baseline")
    print("3. Update your portfolio with results")


if __name__ == "__main__":
    main()