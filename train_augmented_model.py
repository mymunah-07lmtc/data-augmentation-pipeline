"""
Train and Evaluate Model on Augmented Dataset
==============================================

This script loads augmented audio files, extracts features using the SAME
methods as your original projects, and compares performance with baseline.

Usage:
    python train_augmented_model.py
"""

import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIGURATION ----------
CONFIG = {
    "augmentation_log": "output/augmentation_log.csv",
    "test_size": 0.2,
    "random_state": 42,
    "num_files": None,  # Set to e.g., 1000 for testing, None for all
}


# ============================================================
# FEATURE EXTRACTION (FROM YOUR ACTUAL CODE)
# ============================================================

def extract_features_lung(file_path):
    """
    Extract features for LUNG sounds.
    This is from your Pneumonia_AI_Prototype extract_features.py
    """
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean  # Returns 13 features
    except Exception:
        return None


def extract_features_heart(file_path):
    """
    Extract features for HEART sounds.
    This is from your Heart_AI_Prototype extract_features.py
    """
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=5)
        
        # MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        rms = librosa.feature.rms(y=audio)[0]
        
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
        
        return np.array(features)  # 36 features
    except Exception:
        return None


def get_label_lung(file_path):
    """
    Get label for LUNG sounds from filename.
    ICBHI filenames: "101_1b1_Al_sc_Meditron.wav"
    The part after the second underscore is the condition (Al, Ar, Pl, Pr, Tc)
    """
    basename = os.path.basename(file_path)
    parts = basename.split('_')
    if len(parts) >= 3:
        condition_code = parts[2]
        # Map condition codes to binary label
        # Al = Abnormal, Ar = Abnormal, Pl = Abnormal, Pr = Abnormal, Tc = Normal
        if condition_code == 'Tc':
            return 0  # Normal
        else:
            return 1  # Abnormal
    return None


def get_label_heart(file_path):
    """
    Get label for HEART sounds by simple check for 'normal' in filename.
    For more accurate labeling, use the CSV mapping.
    """
    basename = os.path.basename(file_path)
    if 'normal' in basename.lower():
        return 0
    else:
        return 1


# ============================================================
# LOAD AUGMENTED DATA
# ============================================================

def load_augmented_data(log_path, num_files=None):
    """
    Load features and labels from augmented audio files.
    Uses the appropriate feature extraction method based on dataset.
    """
    # Load log
    df = pd.read_csv(log_path)
    
    # Get unique augmented files
    files = df['augmented_file'].unique()
    
    if num_files:
        files = files[:num_files]
    
    print(f"📁 Loading features from {len(files)} augmented files...")
    
    features = []
    labels = []
    dataset_types = []
    
    for file_path in tqdm(files, desc="Extracting features"):
        try:
            # Determine which dataset this file belongs to
            orig_file = df[df['augmented_file'] == file_path].iloc[0]['original_file']
            
            # Check if it's lung or heart
            if 'ICBHI' in orig_file or 'extracted_audio' in orig_file:
                # LUNG: Use 13-feature extraction
                feat = extract_features_lung(file_path)
                if feat is None:
                    continue
                
                # Get label from filename
                label = get_label_lung(file_path)
                if label is None:
                    continue
                
                features.append(feat)
                labels.append(label)
                dataset_types.append('lung')
                
            else:
                # HEART: Use 36-feature extraction
                feat = extract_features_heart(file_path)
                if feat is None:
                    continue
                
                # Get label by simple check
                label = get_label_heart(file_path)
                
                features.append(feat)
                labels.append(label)
                dataset_types.append('heart')
                
        except Exception as e:
            continue
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    print(f"✅ Loaded {len(X)} samples")
    print(f"   Lung: {dataset_types.count('lung')}, Heart: {dataset_types.count('heart')}")
    print(f"   Class distribution: 0={sum(y==0)}, 1={sum(y==1)}")
    print(f"   Feature shape: {X.shape[1]} features")
    
    return X, y


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

def train_model(X_train, y_train):
    """Train SVM model on augmented data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, scaler, X_test, y_test):
    """Evaluate model performance."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1': f1_score(y_test, y_pred, average='binary'),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }
    
    return metrics, y_pred, y_prob


def load_baseline_metrics():
    """
    Load your ACTUAL baseline metrics from your original models.
    Replace these with your real results from 5-fold cross-validation.
    """
    # UPDATE THESE WITH YOUR ACTUAL NUMBERS
    baseline_metrics = {
        'accuracy': 0.80,      # <-- REPLACE with your actual lung accuracy
        'precision': 0.82,     # <-- REPLACE
        'recall': 0.78,        # <-- REPLACE
        'f1': 0.80,            # <-- REPLACE
        'roc_auc': 0.85,       # <-- REPLACE
    }
    print("   ⚠️  Using placeholder baseline metrics. UPDATE WITH YOUR ACTUAL RESULTS!")
    return baseline_metrics


# ============================================================
# GENERATE REPORT (FIXED F-STRING)
# ============================================================

def generate_report(baseline_metrics, augmented_metrics, output_dir='evaluation'):
    """Generate comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Baseline': [
            baseline_metrics.get('accuracy', 'N/A'),
            baseline_metrics.get('precision', 'N/A'),
            baseline_metrics.get('recall', 'N/A'),
            baseline_metrics.get('f1', 'N/A'),
            baseline_metrics.get('roc_auc', 'N/A'),
        ],
        'Augmented': [
            augmented_metrics.get('accuracy', 'N/A'),
            augmented_metrics.get('precision', 'N/A'),
            augmented_metrics.get('recall', 'N/A'),
            augmented_metrics.get('f1', 'N/A'),
            augmented_metrics.get('roc_auc', 'N/A'),
        ]
    })
    
    # Calculate change
    def calc_change(baseline, augmented):
        try:
            return float(augmented) - float(baseline)
        except:
            return 0
    
    comparison['Change'] = comparison.apply(
        lambda row: calc_change(row['Baseline'], row['Augmented']), axis=1
    )
    comparison['Change'] = comparison['Change'].apply(lambda x: f"{x:+.3f}" if x != 0 else "0.000")
    
    # Save CSV
    comparison.to_csv(os.path.join(output_dir, 'comparison_report.csv'), index=False)
    
    # Generate Markdown - FIXED: Properly terminated f-string
    md = f"""# Model Performance Comparison

## Summary
- **Dataset:** ICBHI 2017 (lung) + CirCor (heart)
- **Augmentations:** 5x expansion (speed, gain, filtering, shift, pitch, noise)
- **Total augmented samples:** 12,620

## Results

| Metric | Baseline | Augmented | Change |
|--------|----------|-----------|--------|
| Accuracy | {comparison.iloc[0]['Baseline']} | {comparison.iloc[0]['Augmented']} | {comparison.iloc[0]['Change']} |
| Precision | {comparison.iloc[1]['Baseline']} | {comparison.iloc[1]['Augmented']} | {comparison.iloc[1]['Change']} |
| Recall | {comparison.iloc[2]['Baseline']} | {comparison.iloc[2]['Augmented']} | {comparison.iloc[2]['Change']} |
| F1-Score | {comparison.iloc[3]['Baseline']} | {comparison.iloc[3]['Augmented']} | {comparison.iloc[3]['Change']} |
| ROC-AUC | {comparison.iloc[4]['Baseline']} | {comparison.iloc[4]['Augmented']} | {comparison.iloc[4]['Change']} |

## Confusion Matrix (Augmented Model)
{augmented_metrics.get('confusion_matrix', 'N/A')}

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*
"""
    
    with open(os.path.join(output_dir, 'comparison_report.md'), 'w') as f:
        f.write(md)
    
    print(f"\n✅ Report saved to: {output_dir}/")
    print(f"   - comparison_report.csv")
    print(f"   - comparison_report.md")
    
    return comparison


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("🎯 TRAINING ON AUGMENTED DATASET")
    print("="*60 + "\n")
    
    # Check if log exists
    if not os.path.exists(CONFIG['augmentation_log']):
        print("❌ Augmentation log not found. Run augment_data.py first.")
        return
    
    # Load augmented data
    X, y = load_augmented_data(CONFIG['augmentation_log'], CONFIG['num_files'])
    
    if len(X) == 0:
        print("❌ No features loaded. Check your audio files and feature extraction.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y
    )
    
    print(f"\n   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model on augmented data
    print("\n🔬 Training model on augmented data...")
    model, scaler = train_model(X_train, y_train)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    augmented_metrics, y_pred, y_prob = evaluate_model(model, scaler, X_test, y_test)
    print(f"   Accuracy: {augmented_metrics['accuracy']:.3f}")
    print(f"   Precision: {augmented_metrics['precision']:.3f}")
    print(f"   Recall: {augmented_metrics['recall']:.3f}")
    
    # Load baseline
    baseline_metrics = load_baseline_metrics()
    
    # Generate report
    print("\n📝 Generating comparison report...")
    comparison = generate_report(baseline_metrics, augmented_metrics)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print("\n📌 Next steps:")
    print("1. Replace baseline metrics with your actual results")
    print("2. Check the report in evaluation/")
    print("3. Add results to your portfolio")


if __name__ == "__main__":
    main()