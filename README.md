# 🎵 Data Augmentation Pipeline — Sahel Sound Triage Platform

> A Python pipeline that expands lung sound datasets 5x using realistic augmentations, improving model accuracy from 80% to 84.6%.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Librosa](https://img.shields.io/badge/Librosa-0.10-green.svg)](https://librosa.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)

---

## 📌 Problem

AI models for medical audio diagnosis are often trained on clean, controlled datasets. But in real-world settings — especially in rural Mali — recordings are noisy, inconsistent, and captured with low-quality hardware. This leads to a **performance gap** between research and deployment.

This pipeline bridges that gap by augmenting training data with realistic noise and transformations.

---

## 🎯 What This Pipeline Does

| Augmentation | Purpose |
|--------------|---------|
| Background noise (white/brown/pink) | Simulates rural clinic environments |
| Speed variation (±20%) | Captures different breathing rates |
| Volume variation (±10dB) | Simulates microphone positioning |
| Low-pass filtering | Simulates low-quality hardware |
| Time shift | Handles alignment variation |
| Pitch shift (±2 semitones) | Simulates vocal variation |
| Gaussian noise | Simulates electronic interference |

---

## 📊 Results

| Metric | Baseline (Original) | Augmented | Improvement |
|--------|---------------------|-----------|-------------|
| **Accuracy** | 80.0% | **84.6%** | **+5.7%** |
| Dataset Size | 920 samples | **4,600 samples** | **5x expansion** |
| Best Model | SVM (RBF) | SVM (RBF) | Same model, better performance |

The model was evaluated using **5-fold stratified cross-validation** on the ICBHI 2017 Respiratory Sound Database.

---

## 🛠️ Tech Stack

- **Python** 3.11+
- **Librosa** — Audio processing
- **Audiomentations** — Data augmentation
- **Scikit-learn** — Model training & evaluation
- **SoundFile** — Audio I/O
- **TQDM** — Progress bars

---

## 📁 Repository Structure

```
data_augmentation_pipeline/
├── augment_data.py                # Augmentation script
├── extract_augmented_features.py  # Feature extraction from augmented audio
├── train_augmented_simple.py      # Training and evaluation
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
├── augmented_lung_features.csv    # Extracted features (4,600 samples)
├── augmented_model_best.pkl       # Trained model (SVM, 84.6% accuracy)
├── augmented_scaler.pkl           # Feature scaler
└── output/
    ├── augmented_lung/            # 4,600 augmented audio files
    └── augmentation_log.csv       # Mapping original → augmented
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1. Augment the dataset (skip if you already have augmented files)
python augment_data.py

# 2. Extract features from augmented audio
python extract_augmented_features.py

# 3. Train and evaluate the model
python train_augmented_simple.py
```

### Output

- `augmented_lung_features.csv` — Features for training
- `augmented_model_best.pkl` — Trained model
- `augmented_scaler.pkl` — Feature scaler

---

## 📈 Model Performance Details

| Model | CV Accuracy | Std Dev |
|-------|-------------|---------|
| Random Forest | 83.9% | ±0.29% |
| **SVM (RBF)** | **84.6%** | **±0.56%** |

The SVM model outperformed Random Forest and was selected as the best model.

---

## 🔄 How It Works

1. **Load** audio files from the ICBHI 2017 dataset
2. **Apply** 7 different augmentations to each file
3. **Save** augmented files to `output/augmented_lung/`
4. **Extract** 36 audio features (MFCCs + spectral features)
5. **Train** SVM and Random Forest models with 5-fold CV
6. **Compare** performance with baseline

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👩🏾💻 Author

**Maimouna Tougoutcho Coulibaly**
- GitHub: [@mymunah-07lmtc](https://github.com/mymunah-07lmtc)
- LinkedIn: [maimouna-tougoutcho-coulibaly](https://linkedin.com/in/maimouna-tougoutcho-coulibaly)
- Email: maimounatc@gmail.com

---

## 🙏 Acknowledgements

- **ICBHI 2017** — Respiratory Sound Database
- **Audiomentations** — Augmentation library
- **Librosa** — Audio analysis

---

## 📌 Next Steps

- [ ] Apply augmentation to heart sound dataset (CirCor)
- [ ] Clinical validation with real Malian patient data
- [ ] Deploy augmented model to Raspberry Pi

---

**Built with ❤️ by Maimouna Tougoutcho Coulibaly**

