# 🎧 Deep Learning for Speech Enhancement

This project implements a two-stage deep learning model using **Conv-TasNet** to enhance speech quality in noisy and reverberant environments. Designed to improve intelligibility and clarity, it is particularly useful for applications like ASR systems, hearing aids, and telecommunications.

## 🔍 Overview

Traditional speech enhancement methods fail under real-world conditions due to their assumptions (e.g., stationary noise). Our model directly processes time-domain signals and uses **dilated convolutions** and **dynamic bucket batching** to handle variable-length noisy inputs effectively.

We leverage:

- A **two-stage model** for denoising and dereverberation
- Real-world dataset: **VoiceBank + DEMAND**
- Metrics: **PESQ** (quality), **STOI** (intelligibility)

## 🧠 Model Architecture

- **Base architecture**: Conv-TasNet
- **Stage 1**: Denoising
- **Stage 2**: Dereverberation
- **Activation**: ReLU
- **Loss**: MSE  
- **Optimizer**: Adam

The model generalizes across both matched and mismatched acoustic conditions.

## 📁 Folder Structure

```
.
├── main.py               # Full training and evaluation script
├── /content/unzipped_files/
│   ├── clean_trainset_28spk_wav/
│   ├── noisy_trainset_28spk_wav/
│   ├── clean_testset_wav/
│   └── noisy_testset_wav/
```

## 🗃 Dataset

We use the [VoiceBank-DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/2791):

- **28 speakers** (14 male, 14 female)
- Clean and noisy paired samples
- Variety of indoor, outdoor, and home noise types
- Preprocessed to 16 kHz sampling rate

## 📈 Evaluation Metrics

- **PESQ**: Perceptual Evaluation of Speech Quality (range -0.5 to 4.5)
- **STOI**: Short-Time Objective Intelligibility (range 0 to 1)

### Matched Conditions:
- Noisy PESQ: ~1.95 → Enhanced PESQ: ~3.25
- Noisy STOI: ~0.72 → Enhanced STOI: ~0.92

### Mismatched Conditions:
- Noisy PESQ: ~1.80 → Enhanced PESQ: ~3.10
- Noisy STOI: ~0.68 → Enhanced STOI: ~0.89

## 📊 Visualizations

- Training vs Validation Loss curves
- Bar plots comparing PESQ and STOI for matched and mismatched scenarios

## 🔧 Requirements

Install dependencies:

```bash
pip install tensorflow librosa pesq pystoi matplotlib scikit-learn
```

## ▶️ Running the Project

Make sure dataset paths in `main.py` are correct, then run:

```bash
python main.py
```

The script will:
- Load and augment data
- Train the two-stage model
- Evaluate using PESQ and STOI
- Plot loss curves and bar charts

## 🧩 Future Work

- Explore transformer-based architectures
- Add support for real-time processing
- Optimize model for mobile/edge deployment (via quantization or pruning)
- Integrate visual modalities for multi-modal speech enhancement

## 📜 Citation

If you use this work, please cite:

> Shreyas Srinivas Bikumalla, Amith Reddy Atla. *Enhancing Speech Quality in Noisy Environments*, 2024.
