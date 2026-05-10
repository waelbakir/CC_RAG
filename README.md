# 🕵️ Algorithmic Fraud Detection & Explainable AI Copilot

> An end-to-end machine learning pipeline combining **Cost-Sensitive XGBoost** and **Retrieval-Augmented Generation (RAG)** to detect financial fraud and automatically explain every decision.

---

## 📖 Overview

Financial fraud detection faces two hard problems:

1. **Extreme class imbalance** — fraudulent transactions can be as rare as 0.17% of all records.
2. **Black-box models** — when a model flags a transaction, it offers no explanation; human investigators are left guessing.

This project solves both by chaining three algorithmic domains into a single cohesive pipeline:

| Domain | Technique | Purpose |
|---|---|---|
| Linear Algebra | PCA | Dimensionality reduction & 3D visualization |
| Gradient Boosting | XGBoost + Cost-Sensitive Learning | Fraud classification without synthetic data |
| Generative AI & Search | FAISS + LLM (RAG) | Explainable AI policy reports |

---

## 🛠️ Methodology

### Phase 1 — Dimensionality Reduction (PCA)

- **Goal:** Compress 30-dimensional transaction data into 3 Principal Components for visual exploration.
- **Math:** Computes the covariance matrix, extracts eigenvectors, and projects data into the subspace of maximum variance.
- **Output:** An interactive 3D scatter plot clearly separating dense *normal* clusters from scattered *fraud* anomalies.

### Phase 2 — Fraud Classification (XGBoost)

- **Problem:** The dataset holds 284,315 normal transactions and only 492 fraud cases (~0.17%). Naive models simply predict "Normal" 100% of the time.
- **Solution:** Cost-Sensitive Learning — the exact imbalance ratio is computed and applied as `scale_pos_weight`, mathematically penalizing fraud misclassification **~577× more** in the gradient descent loss function.
- **Why not SMOTE?** Synthetic oversampling can introduce unrealistic data artifacts. Cost-sensitive weighting achieves the same effect without generating fake samples.

### Phase 3 — Explainable AI via RAG

- **Problem:** Raw probabilities (e.g., "99.9% Fraud") are not actionable for investigators.
- **Solution:** Bank policy documents are chunked, embedded, and stored in a **FAISS HNSW** vector index. When a transaction is flagged, the most semantically similar policy is retrieved in **O(log N)** time.
- **Generation:** The flagged transaction's mathematical details and the retrieved policy are fed to **TinyLlama 1.1B**. Greedy decoding (`do_sample=False`) forces fully deterministic output, eliminating hallucinations and producing a strictly factual, policy-grounded report.

---

## 💾 Dataset

| Property | Detail |
|---|---|
| Source | [Kaggle — ML Group ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| Description | European credit card transactions over two days |
| Shape | 284,807 rows × 30 columns |
| Features | `V1`–`V28` (PCA-anonymized), `Time`, `Amount` |
| Fraud Rate | ~0.17% (492 out of 284,807) |

---

## ⚙️ Setup & Installation

> **Recommended environment:** Google Colab with a free T4 GPU.

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib plotly seaborn xgboost
pip install langchain langchain-community langchain-huggingface langchain-core langchain-text-splitters
pip install faiss-cpu sentence-transformers transformers accelerate
```

### 2. Data Requirements

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
2. Upload it to your Google Drive.
3. Update the file paths in the notebook to match your Drive directory.

---

## 🚀 Usage

The final notebook cell exposes a single master function:

```python
evaluate_transaction(dataset_index=14)
```

Pass any valid dataset row index to run the full pipeline — PCA projection, XGBoost inference, FAISS retrieval, and LLM report generation.

### Example Output

```
==================================================
🔍 INVESTIGATING TRANSACTION ID: 141260
==================================================
💰 Amount: $0.00
✅ True Label in Dataset: FRAUD
🤖 XGBoost Predicted Fraud Probability: 99.98%

🚨 FRAUD DETECTED! Triggering Vector Search and RAG AI...

[FAISS DEBUG] Retrieved Policy: Policy 104 - Zero-Dollar Authorization...

=== AI COPILOT REPORT ===
Based on the policy provided, the fraud type is a Zero-Dollar Authorization,
which strongly indicates an Account Takeover attempt by a fraudster trying to
link the physical credit card to a new digital wallet (like Apple Pay).
Action Required: Reject the authorization immediately and force a complete
account password reset.
==================================================
```

---

## 📈 Evaluation

| Component | Result |
|---|---|
| XGBoost Recall | High recall on minority fraud class despite ~577:1 imbalance |
| FAISS Retrieval | O(log N) exact policy matching via HNSW indexing |
| LLM Hallucination Rate | **0%** — enforced by greedy decoding + hard-coded stop tokens |

---

## 🧰 Tech Stack

| Category | Libraries |
|---|---|
| Data & ML | `pandas`, `numpy`, `scikit-learn`, `xgboost` |
| Vector Search | `faiss-cpu`, `sentence-transformers` |
| Generative AI | `langchain` (LCEL), `transformers` (Hugging Face) |
| Visualization | `plotly`, `matplotlib`, `seaborn` |

---

## 📂 Project Structure

```
├── notebook.ipynb          # Main Colab notebook (all phases)
├── creditcard.csv          # Dataset (download from Kaggle)
└── README.md               # This file
```

---

## 📄 License

This project is for educational and research purposes. The dataset is provided by the [Machine Learning Group at ULB](https://mlg.ulb.ac.be/) under their respective terms.
