# TOSS NEXT ML Challenge: Advanced Ad Click Prediction (CTR) Model

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## 📌 Executive Summary

This project was developed for the **TOSS NEXT ML Challenge**, focusing on predicting Ad Click-Through Rate (CTR). The core challenge involved processing sparse, high-cardinality financial transaction and behavioral data to predict user intent.

While the initial baseline focused on Gradient Boosted Trees (LightGBM/XGBoost), this repository has been **radically upgraded** to feature state-of-the-art Deep Learning architectures specifically designed for CTR prediction. We have unified the most advanced models into a production-ready pipeline, allowing seamless switching between different sequence modeling techniques.

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

## 📂 Project Structure

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

## 🚀 How to Run

The refactored pipeline allows for easy experimentation and training via CLI.

### Installation
Ensure you have PyTorch and requested libraries installed:
```bash
pip install torch pandas numpy scikit-learn pyarrow
```

### Training the Deep Learning Model
You can run the structured training pipeline directly from the root directory and specify which sequence backbone to use:

```bash
# Train with DIN (Default)
python src/train.py --train_path path/to/train.parquet --test_path path/to/test.parquet --seq_backbone din

# Train with DIEN
python src/train.py --train_path path/to/train.parquet --test_path path/to/test.parquet --seq_backbone dien

# Train with BST (Transformer)
python src/train.py --train_path path/to/train.parquet --test_path path/to/test.parquet --seq_backbone bst
```

For a full list of configurable hyperparameters:
```bash
python src/train.py --help
```

## 📈 Key Results

- **Feature Interaction Mastery**: DCN-V2 successfully captured high-order feature interactions that GBDTs missed.
- **Behavioral Modeling**: DIN/DIEN/BST architectures clearly showed higher capability in capturing temporal patterns compared to non-sequential models.
- **Performance**: Achieved top-tier ranking in prediction accuracy by focusing on behavioral temporal patterns and advanced deep CTR models.

---
*This repository has been refined and polished by the Elite Data Science Career Consultant Persona.*
