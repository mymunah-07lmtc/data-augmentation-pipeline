# 🎵 Data Augmentation Pipeline — Sahel Sound Triage Platform

> Realistic audio augmentation pipeline for medical AI — expanding the ICBHI 2017 respiratory dataset by 5x with noise, speed, gain, filtering, and pitch variations. Improved lung sound classifier accuracy from **80% to 84.6%**.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Librosa](https://img.shields.io/badge/Librosa-0.10-orange.svg)](https://librosa.org/)
[![Audiomentations](https://img.shields.io/badge/Audiomentations-0.30-green.svg)](https://github.com/iver56/audiomentations)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem

AI models for medical audio classification are often trained on clean, controlled datasets (ICBHI 2017, CirCor). However, real-world deployment in rural Mali presents challenges:
- **Background noise** — motorcycles, markets, wind
- **Power-line interference** — 50Hz hum from unstable electricity
- **Poor microphone quality** — cheap hardware in the field
- **Variable recording conditions** — different mic positions, breathing rates

This pipeline bridges the gap by augmenting training data with realistic variations, improving model robustness.

---

## ✨ Features

### Augmentations Applied

| Augmentation | Description | Real-World Simulation |
| :--- | :--- | :--- |
| **Background Noise** | White, brown, pink noise | Motorcycles, markets, wind |
| **Speed Variation** | 0.8x – 1.2x | Different breathing rates |
| **Volume Variation** | ±10dB | Microphone placement |
| **Filtering** | High-pass filter (50–150Hz) | Low-quality microphones |
| **Time Shift** | ±0.2 seconds | Alignment variation |
| **Pitch Shift** | ±2 semitones | Vocal variation |
| **Gaussian Noise** | Electronic interference | Power-line hum |

### Results

| Metric | Baseline | Augmented | Improvement |
|--------|----------|-----------|-------------|
| **Accuracy** | 80.0% | **84.6%** | **+5.7%** |
| **Dataset Size** | 920 | **4,600** | **5x expansion** |
| **Best Model** | — | SVM (RBF) | 84.6% CV accuracy |

---

## 🛠️ Tech Stack

- **Audio Processing:** `librosa`, `soundfile`
- **Augmentation:** `audiomentations`
- **Machine Learning:** `scikit-learn` (SVM, Random Forest)
- **Data Handling:** `pandas`, `numpy`
- **Progress Tracking:** `tqdm`

---

## 📁 Project Structure
data_augmentation_pipeline/
├── augment_data.py # Augmentation script (generates 5x data)
├── extract_augmented_features.py # Feature extraction from augmented audio
├── train_augmented_simple.py # Training on augmented features
├── icbhi_diagnosis.csv # Metadata for labels
├── requirements.txt # Python dependencies
├── README.md # This file
├── .gitignore # Git ignore rules
├── augmented_lung_features.csv # Extracted features (4,600 samples)
├── augmented_model_best.pkl # Best model (SVM, 84.6% accuracy)
├── augmented_scaler.pkl # Feature scaler
└── output/
├── augmented_lung/ # 4,600 augmented .wav files
└── augmentation_log.csv # Mapping original → augmented

text

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/mymunah-07lmtc/data-augmentation-pipeline.git
cd data-augmentation-pipeline

# Install dependencies
pip install -r requirements.txt
Run the Pipeline
bash
# 1. Augment audio files (generates 5x data)
python augment_data.py

# 2. Extract features from augmented audio
python extract_augmented_features.py

# 3. Train model on augmented data
python train_augmented_simple.py
📊 Results Breakdown
Cross-Validation Performance
Model	CV Accuracy	Std Dev
Random Forest	83.85%	±0.29%
SVM (RBF)	84.59%	±0.56%
Class Distribution
Class	Count	Percentage
Normal (0)	765	16.6%
Abnormal (1)	3,835	83.4%
Improvement Summary
The augmentation pipeline improved model accuracy by 5.7% over the baseline (80.0% → 84.6%).

🔍 Example Augmentation
Original Audio:

text
[Clean lung sound]
Augmented Audio:

text
[Lung sound + Background noise + Speed variation + Gain change]
The augmented audio simulates what a community health worker would hear in a rural clinic in Mali.

📈 Next Steps
Clinical validation — Test the augmented model with real Malian patient data

Apply to heart sounds — Extend the pipeline to the CirCor dataset

Deploy to Raspberry Pi — Integrate the augmented model into the field device

Publish the augmented dataset — Share on Hugging Face or Zenodo

🤝 Contributing
Contributions are welcome! Please:

Fork the repository

Create a feature branch

Submit a pull request

📝 License
This project is licensed under the MIT License — see the LICENSE file for details.

👩🏾‍💻 Author
Maimouna Tougoutcho Coulibaly

GitHub: @mymunah-07lmtc

LinkedIn: maimouna-tougoutcho-coulibaly

Email: maimounatc@gmail.com

🙏 Acknowledgements
ICBHI 2017 Respiratory Sound Database

Librosa — Audio processing

Audiomentations — Augmentation library

Built with ❤️ by Maimouna Tougoutcho Coulibaly