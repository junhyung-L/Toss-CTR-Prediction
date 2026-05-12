# 🖱️ TOSS NEXT ML CHALLENGE: Ad Click Prediction (CTR) Model

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

## 🚀 Executive Summary (TL;DR)
- **The Problem**: Predict Ad Click-Through Rate (CTR) for Toss app users, handling extreme class imbalance (pos_weight ~51.4), massive sequence data, and memory constraints.
- **The Solution**: Developed an advanced hybrid deep learning model combining **DCN-V2 (CrossNetMix)** for explicit feature interactions and **DIN (Deep Interest Network)** with a custom activation unit for user behavior sequence modeling.
- **The Result**: Achieved a validation AUC of **0.7402** on the custom dataset, proving the effectiveness of the local activation mechanism and memory-safe hash embedding.

## 🛠 Tech Stack
- **Framework**: PyTorch 2.x
- **Modeling**: DCN-V2 + DIN (Custom Hybrid)
- **Sequence Processing**: Hash Embedding (262,144 buckets)
- **Optimization**: AdamW, Cosine Annealing, Mixed Precision (AMP)

---

## 🔬 1. Problem Definition
Predicting whether a user will click on an advertisement (Click-Through Rate or CTR) is the core engine of digital marketing, directly impacting revenue and user experience.
- **The Challenge**: Predict Ad CTR for Toss app users.
- **The Complications**: Extreme class imbalance (very few users click on ads compared to those who don't), massive sequence data representing user behavior, and strict memory constraints for production deployment.
- **Objective**: To build a high-performance deep learning model that accurately predicts CTR while remaining memory-efficient.

---

## 🛠️ 2. System Architecture: DCN-V2 + DIN
To capture both cross-feature interactions and the evolution of user interests, we fused two state-of-the-art architectures. This hybrid approach ensures we model both static user profiles and dynamic behavior.

```mermaid
graph TD
    A[Input Features] --> B[Continuous Features]
    A --> C[Categorical Features]
    A --> D[User Behavior Sequence]
    
    B --> E[BatchNorm1d]
    C --> F[Embedding Lookup]
    D --> G[Hash Embedding <br> 2^18 Buckets]
    
    E --> H[Unified Tabular Vector]
    F --> H
    
    H --> I[DCN-V2 <br> CrossNetMix]
    H --> J[Deep Tower <br> 512-256-128]
    
    G --> K[DIN Attention <br> Target x History]
    
    I --> L[Concatenation]
    J --> L
    K --> L
    
    L --> M[Prediction Head]
    M --> N[CTR Output]
```

---

## 🤖 3. Deep Dive: Modeling Details
To solve the problem, we implemented advanced deep learning techniques to handle both feature interactions and sequence modeling.

### 🧠 A. DCN-V2 (CrossNetMix)
Instead of standard feature crosses, which explode in memory with high-dimensional data, we implemented the memory-efficient **CrossNetMix** using Einstein summation (`torch.einsum`):
- **Mechanism**: Low-rank approximation (`rank=32`) with 4 experts to model complex non-linear feature interactions without memory explosion.

### 📍 B. DIN (Deep Interest Network) with Custom Activation
To model the user's dynamic interests relative to the target ad (e.g., a user who likes finance might click a loan ad but ignore a game ad):
- **Local Activation Unit**: Calculates attention weights between the target ad and historical ad behaviors.
- **Features for Attention**: `[q, k_i, q - k_i, q * k_i]` fed into a sub-MLP to learn non-linear relationships.

### ⚙️ C. Big Data Scale & Optimization
- **Hash Embedding**: Used a fixed bucket size of **262,144** for sequence items to prevent Out-Of-Memory (OOM) errors caused by high cardinality.
- **Imbalance Handling**: Used `BCEWithLogitsLoss` with a calculated `pos_weight` of ~51.4 to force the model to learn from the rare positive click events.

### 📊 Experiment Benchmarks
We prototyped and evaluated several state-of-the-art CTR models to find the best approach:
- **CatBoost + DIN**: Validation AUC **0.7412**, PR-AUC **0.0779** (Best performing hybrid approach).
- **DCN-V2 + DIN**: Validation AUC **0.7402**, PR-AUC **0.0775** (Deep learning focused).

---

## 🏁 4. Conclusion & Business Impact
The project successfully demonstrated how to build a production-ready CTR model under extreme constraints.
- **Outcome**: Achieved a validation AUC of **0.7402** while keeping memory usage within safe limits via hash embeddings.
- **Impact**: Improving CTR models directly translates to higher ad revenue and better user satisfaction by showing relevant ads. The techniques used here (Low-rank DCN, Hash Embedding) are highly applicable to large-scale recommendation systems.

---

## 📁 Repository Structure
```text
├── notebooks/                  # Experimental Notebooks
│   ├── baseline.ipynb
│   ├── DCN-V2 + DIN.ipynb     # Main Model (This Document)
│   ├── DCN-V2 + (DIEN or BST).ipynb
│   └── dusin(full).ipynb
├── prior-research/             # Background research and papers
├── results/                    # Saved models and predictions
├── src/                        # Production-Ready Source Code
│   ├── data.py                 # Data loading and preprocessing
│   ├── models.py               # DCN-V2 & DIN model definitions
│   └── train.py                # Training loop and evaluation
└── main.py                     # Master pipeline runner
```

## ⚙️ How to Run
1. Install dependencies:
   ```bash
   pip install torch pandas numpy scikit-learn
   ```
2. Run the main notebook:
   - Open `notebooks/DCN-V2 + DIN.ipynb` and execute cells. It includes dummy data generation if the raw parquet files are missing.

## 👥 Contributors
- **Junhyung L.** (Project Lead)

---
*Refactored and polished to meet professional software engineering standards for the [Data Analyst Portfolio](https://github.com/junhyung-L).*
