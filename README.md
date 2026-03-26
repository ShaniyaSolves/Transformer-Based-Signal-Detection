# Transformer-Based Signal Detection for AGN Light Curves

This project applies a **pretrained Whisper transformer (audio model)** to classify astrophysical time-series data. The goal is to detect periodic signals from **binary supermassive black hole (SMBH) systems** embedded in noisy AGN light curves. It includes physics-based data generation, signal analysis (FAP/SNR), and parameter-efficient fine-tuning (LoRA/DoRA).

---

## Main Idea
Astrophysical light curves show the variation of an object's brightness overtime. When a binary black hole system is present, their orbital motion creates a subtle, periodic signal burried within that light curve. In this work, I treat the data like an **audio signals**, allowing the use of a pretrained speech transformer (Whisper) for pattern recognition in noisy environments.

---

## Methods
- Simulated AGN light curves using:
  - `stingray` (red noise variability)
  - `binlite` (binary accretion + flux modeling)
- Converted time series into Whisper-compatible input features
- Fine-tuned the Whisper encoder using **LoRA / DoRA** (parameter-efficient training)
- Binary classification task:
  - 0 = noise-only AGN  
  - 1 = binary SMBH signal + noise AGN  

---

## 📊 Current Status
- End-to-end pipeline implemented (simulation → preprocessing → model → training)
- Initial experiments suggest performance depends strongly on signal strength (SNR)
- Currently working to improve detection in low-SNR regimes

---

## 🛠️ Tech Stack
- PyTorch  
- HuggingFace Transformers (Whisper)  
- PEFT (LoRA / DoRA)  
- Stingray  
- BinLite  

---

## How to Run
pip install -r requirements.txt
python main_pipeline.py
