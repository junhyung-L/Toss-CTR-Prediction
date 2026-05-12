# TOSS NEXT ML Challenge: Advanced Ad Click Prediction (CTR) Model

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## 📌 Executive Summary

This project was developed for the **TOSS NEXT ML Challenge**, focusing on predicting Ad Click-Through Rate (CTR). The core challenge involved processing sparse, high-cardinality financial transaction and behavioral data to predict user intent.

## 🚀 Executive Summary (TL;DR)
- **The Problem**: Predicting Ad CTR using sparse, high-cardinality financial and behavioral data.
- **The Solution**: Upgraded from standard GBDT to state-of-the-art Deep Learning architectures (DCN-V2) with unified sequence backbones (DIN, DIEN, BST).
- **The Result**: Captured complex high-order feature interactions and temporal behavior patterns, achieving top-tier ranking.

## 📊 Data & Preprocessing
- **Sequence Parsing**: Extracted and aligned historical user behavior sequences to capture evolving interests.
- **High Cardinality**: Applied **Hash Embedding** to handle massive categorical feature spaces without OOM issues.
- **Class Imbalance**: Addressed extreme class imbalance using custom `pos_weight` in Binary Cross-Entropy loss.

## 🛠️ Methodologies & Advanced Architectures

We transitioned from standard GBDT models to advanced neural architectures to better handle feature interactions and sequential behaviors. The repository now supports a unified **DCN-V2 (Deep & Cross Network v2)** body with multiple sequence backbones:

### 1. Deep & Cross Network v2 (DCN-V2)
- **CrossNetMix**: Utilizes low-rank approximation and mixture-of-experts to learn explicit feature interactions of arbitrary orders efficiently.
- Combined with a Deep MLP tower to capture implicit non-linear interactions.

### 2. Supported Sequence Backbones
You can choose the sequence backbone that best fits the data behavior:
- **DIN (Deep Interest Network)**: Implements a local activation unit (attention mechanism) to adaptively learn the representation of user interests from historical behaviors w.r.t. a specific candidate ad.
- **DIEN (Deep Interest Evolution Network)**: Adds a GRU layer to capture the temporal evolution of user interests before applying the attention mechanism.
- **BST (Behavior Sequence Transformer)**: Leverages the powerful self-attention mechanism of Transformers to capture complex correlations among user behaviors.

### 3. Advanced Techniques
- **Hash Embedding**: Handled high-cardinality sequence features safely without OOM (Out of Memory) issues.
- **Precision Optimization**: Implemented FP16 mixed precision training with safety checks for overflow and NaN values.
- **Imbalance Handling**: Utilized custom `pos_weight` in BCE loss to handle extreme class imbalance.

## 📈 Key Results
- **Feature Interaction Mastery**: DCN-V2 successfully captured high-order feature interactions that GBDTs missed.
- **Behavioral Modeling**: DIN/DIEN/BST architectures clearly showed higher capability in capturing temporal patterns compared to non-sequential models.
- **Performance**: Achieved top-tier ranking in prediction accuracy by focusing on behavioral temporal patterns and advanced deep CTR models.

## 📁 Repository Structure

To ensure production readiness and maturity, the repository has been refactored into a structured pipeline:

```text
├── notebooks/                  # Experimental Jupyter Notebooks
│   ├── baseline.ipynb          # Initial baseline & EDA
│   ├── DCN-V2 + DIN.ipynb      # DCN-V2 combined with DIN
│   ├── DCN-V2 + (DIEN or BST).ipynb # DIEN and BST experiments
│   └── ...                     # Other architecture experiments
├── src/                        # Production-ready Source Code
│   ├── data.py                 # Dataset & Sequence Parsing utilities
│   ├── models.py               # PyTorch implementations of DCN-V2, DIN, DIEN, BST
│   └── train.py                # Parameterized training & inference pipeline
├── results/                    # Prediction results and model meta-data
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules for data and large files
```

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install torch pandas numpy scikit-learn pyarrow
   ```
2. Train the Deep Learning Model:
   ```bash
   # Train with DIN (Default)
   python src/train.py --train_path path/to/train.parquet --test_path path/to/test.parquet --seq_backbone din
   ```

## 👥 Contributors
- **Junhyung L.** (Project Lead)

---
*Refactored and polished to meet professional software engineering standards for the [Data Analyst Portfolio](https://github.com/junhyung-L/Resume/blob/main/Portfolio/README.md).*
