
# 🎵 Sahel Sound Triage — Data Augmentation Pipeline

A robust data augmentation pipeline for the Sahel Sound Triage Platform, designed to improve model performance by simulating real-world acoustic conditions in rural Malian clinics.

- **5x dataset expansion** — from 920 to 4,600 lung sounds, 1,604 to 8,020 heart sounds
- **7 realistic augmentations** — background noise (rural clinics), speed variation, volume changes, low-pass filtering, time shift, pitch shift, and Gaussian noise
- **Proven improvement** — lung model accuracy increased from **80.0% to 94.3%** (+17.9%)
- **Heart model improvement** — F1-Macro increased from **0.5865 to 0.6509** (+11.0%)

---

## 📦 Project Structure

```
data_augmentation_pipeline/
├── augment_data.py                          # Main augmentation script
├── extract_augmented_features.py            # Feature extraction from augmented files
├── train_augmented_model.py                 # Training on augmented dataset
├── train_augmented_simple.py                # Simplified training script
├── retrain_heart_with_age_augmented.py      # Heart model with Age + Augmentation
├── output/
│   ├── augmented_lung/                      # 4,600 augmented lung sounds
│   ├── augmented_heart/                     # 8,020 augmented heart sounds
│   └── augmentation_log.csv                 # Mapping original → augmented files
├── augmented_lung_features.csv              # Lung features from augmented data
├── augmented_heart_features.csv             # Heart features from augmented data
├── augmented_features_all.csv               # Combined features (12,620 samples)
├── augmented_features_all_with_labels.csv   # Combined features + labels
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## 🧠 Why Data Augmentation Matters

The ICBHI 2017 (lung) and CirCor DigiScope (heart) datasets are collected under controlled, quiet conditions. However, rural Malian clinics are noisy environments with:

- Background conversations
- Wind and environmental noise
- Low-quality recording devices
- Variable patient positioning

Training on clean data and deploying in noisy environments leads to **performance degradation**. Data augmentation bridges this gap by teaching the model to be robust to these real-world conditions before it ever leaves the lab.

---

## 🔧 Augmentation Techniques

| Augmentation | Description | Clinical Rationale |
| :--- | :--- | :--- |
| **Background Noise** | Adds ambient clinic noise (recorded separately) | Simulates real-world recording conditions |
| **Speed Variation** | Changes playback speed (±10%) | Accounts for different breathing rates |
| **Volume Variation** | Adjusts amplitude (±10 dB) | Accounts for microphone placement differences |
| **Low-Pass Filtering** | Simulates cheap hardware (cutoff at 2-4 kHz) | Lower-cost stethoscope microphones have limited bandwidth |
| **Time Shift** | Randomly shifts the signal (±0.2s) | Accounts for variable recording start times |
| **Pitch Shift** | Changes pitch (±2 semitones) | Simulates anatomical differences |
| **Gaussian Noise** | Adds subtle random noise | Simulates electronic interference |

---

## 📊 Results

### Lung Model (ICBHI 2017)

| Metric | Baseline | After Augmentation | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 80.0% | **94.3%** | **+17.9%** |
| **Training Samples** | 920 | **4,600** | 5x larger |
| **Cross-Validation** | 5-Fold | **5-Fold** | — |

### Heart Model (CirCor DigiScope)

| Metric | Baseline (Age-Aware) | After Augmentation + Age | Improvement |
| :--- | :--- | :--- | :--- |
| **F1-Macro** | 0.5865 | **0.6509** | **+11.0%** |
| **Normal Recall** | Very low | **0.48** | Significant |
| **Test Accuracy** | ~95% (deceptive) | **92.2%** | More honest |
| **Training Samples** | 1,604 | **8,020** | 5x larger |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mymunah-07lmtc/data-augmentation-pipeline.git
cd data-augmentation-pipeline
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Run the Augmentation Pipeline
```bash
python augment_data.py
```

This will:
- Load original lung (920) and heart (1,604) sounds.
- Apply 7 augmentations to each file.
- Save augmented files to `output/augmented_lung/` and `output/augmented_heart/`.
- Generate `output/augmentation_log.csv`.

### 4. Extract Features from Augmented Data
```bash
python extract_augmented_features.py
```

### 5. Train the Model
```bash
python train_augmented_simple.py
```

### 6. Retrain Heart Model with Age
```bash
python retrain_heart_with_age_augmented.py
```

---

## 📈 Visual Results

### Lung Model Accuracy Before and After Augmentation

```
Accuracy (%)
100 |                                    ⬆ 94.3%
 90 |                                ⬆
 80 |  ⬆ 80.0%
 70 |
 60 |
     Baseline    Augmented
```

### Heart Model F1-Macro Before and After

```
F1-Macro
0.65 |                                    ⬆ 0.6509
0.60 |                                ⬆
0.55 |  ⬆ 0.5865
0.50 |
     Baseline    Augmented + Age
```

---

## 📁 Output Files

| File | Description |
| :--- | :--- |
| `output/augmented_lung/*.wav` | 4,600 augmented lung sounds (5x original) |
| `output/augmented_heart/*.wav` | 8,020 augmented heart sounds (5x original) |
| `output/augmentation_log.csv` | Mapping original → augmented files |
| `augmented_features_all_with_labels.csv` | 12,620 samples with 36 features + labels |

---

## 🔬 Performance Comparison Summary

| Model | Dataset Size | Accuracy / F1 | Status |
| :--- | :--- | :--- | :--- |
| **Lung (Baseline)** | 920 | 80.0% | ❌ |
| **Lung (Augmented)** | 4,600 | **94.3%** | ✅ Deployed |
| **Heart (Baseline)** | 1,604 | F1 0.5865 | ❌ |
| **Heart (Augmented + Age)** | 8,020 | **F1 0.6509** | ✅ Deployed |

---

## 🛠️ Requirements

```txt
librosa==0.11.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.2
tqdm==4.65.0
soundfile==0.12.1
matplotlib==3.7.2
```

---

## 📄 Disclaimer

⚠️ **For Research Purposes Only.** This pipeline is part of a proof-of-concept prototype. It is **NOT** a certified medical device. All models trained with this pipeline must be clinically validated before deployment.

---

## 📬 Contact

**Author:** Maimouna Tougoutcho Coulibaly  
**Email:** maimounatcoul@gmail.com  
**GitHub:** [github.com/mymunah-07lmtc](https://github.com/mymunah-07lmtc)  
**LinkedIn:** [linkedin.com/in/maimouna-tougoutcho-coulibaly](https://linkedin.com/in/maimouna-tougoutcho-coulibaly)

---

**Built with ❤️ in Bamako, Mali | ICBHI 2017 + CirCor DigiScope**
