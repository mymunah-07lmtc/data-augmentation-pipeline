"""
Train Model on Augmented Data
=============================

This script trains your model using the augmented features CSV.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ---------- LOAD AUGMENTED FEATURES ----------
print("\n" + "="*60)
print("🎯 TRAINING ON AUGMENTED DATA")
print("="*60 + "\n")

df = pd.read_csv("augmented_lung_features.csv")
print(f"✅ Loaded {len(df)} augmented samples")

# ---------- LOAD METADATA (for labels) ----------
# Copy your icbhi_diagnosis.csv to this folder first!
try:
    metadata = pd.read_csv("icbhi_diagnosis.csv")
    print("✅ Loaded metadata")
    
    # Merge on Patient_ID
    df = df.merge(metadata, on='Patient_ID', how='left')
    
    # Find diagnosis column
    diag_col = None
    for col in metadata.columns:
        if 'diagnosis' in col.lower() or 'disease' in col.lower() or 'label' in col.lower():
            diag_col = col
            break
    
    if diag_col is None:
        raise ValueError("Could not find diagnosis column")
    
    # Create binary label
    def map_to_binary(diagnosis):
        diagnosis = str(diagnosis).strip().lower()
        if diagnosis in ['normal', 'healthy', 'control']:
            return 0
        else:
            return 1
    
    df['Binary_Label'] = df[diag_col].apply(map_to_binary)
    df = df.dropna(subset=['Binary_Label'])
    
    print(f"✅ Label distribution: {df['Binary_Label'].value_counts().to_dict()}")
    
except Exception as e:
    print(f"⚠️  Could not merge labels: {e}")
    print("   Using labels from filenames instead...")
    
    def label_from_filename(fname):
        # If the original file had 'normal' in the path, it's class 0
        # Your augmented files keep the same base name
        if 'normal' in fname.lower() or 'healthy' in fname.lower():
            return 0
        else:
            return 1
    
    df['Binary_Label'] = df['Filename'].apply(label_from_filename)

# ---------- PREPARE DATA ----------
feature_cols = [col for col in df.columns if col.startswith('MFCC_') or col in [
    'spec_cent_mean', 'spec_cent_std', 'spec_bw_mean', 'spec_bw_std',
    'spec_rolloff_mean', 'spec_rolloff_std', 'zcr_mean', 'zcr_std',
    'rms_mean', 'rms_std'
]]

X = df[feature_cols].values
y = df['Binary_Label'].values

print(f"\n📊 Dataset:")
print(f"   Samples: {len(X)}")
print(f"   Features: {X.shape[1]}")
print(f"   Normal (0): {sum(y==0)}")
print(f"   Abnormal (1): {sum(y==1)}")

# ---------- TRAIN ----------
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n🔬 5-Fold Cross-Validation:")
best_score = -1
best_model = None
best_scaler = None

for name, model in models.items():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    print(f"   {name}: {mean_score:.4f} (+/- {scores.std():.4f})")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_scaler = scaler

# ---------- TRAIN FINAL MODEL ----------
print(f"\n🏆 Best model: {type(best_model).__name__} (CV Acc: {best_score:.4f})")

# Fit on full dataset
X_scaled_full = best_scaler.fit_transform(X)
best_model.fit(X_scaled_full, y)

# ---------- SAVE ----------
joblib.dump(best_model, "augmented_model_best.pkl")
joblib.dump(best_scaler, "augmented_scaler.pkl")

print("\n✅ Saved:")
print("   - augmented_model_best.pkl")
print("   - augmented_scaler.pkl")

# ---------- COMPARE WITH BASELINE ----------
print(f"\n📊 Performance Comparison:")
print(f"   Baseline (original model): ~0.80 (from your documentation)")
print(f"   Augmented model:          {best_score:.4f}")

improvement = (best_score - 0.80) / 0.80 * 100
if improvement > 0:
    print(f"   Improvement: +{improvement:.1f}%")
else:
    print(f"   Change: {improvement:.1f}%")

print("\n✅ Done!")