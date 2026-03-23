# Multiagent Commute AI

A production-style **multi-agent AI pipeline** for automated commute-claim analysis, policy Q&A, and fraud/anomaly detection — built with LangGraph, RAG (FAISS + fine-tuned Sentence Transformers), Isolation Forest, Gradient Boosting, SHAP, and a local Llama 3.2 LLM via Ollama.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph StateGraph            │
│                                             │
│  Intent Agent ──► Policy Agent ──► Anomaly Agent
│       │                │                   │
│  (classifies)    (RAG + LLM +         (Isolation Forest
│                  Query Rewrite)        + Gradient Boost
│                       │                + SHAP)
│                       ▼                   │
│              Explain Agent ◄───────────────┘
│                       │
│                       ▼
│              Synth Agent (final response)
└─────────────────────────────────────────────┘
```

**Agents**
| Agent | Role |
|---|---|
| Intent | Classifies query as `policy_query`, `anomaly_check`, or `general` |
| Policy | RAG retrieval → LLM Query Rewriting → Llama 3.2 answer |
| Anomaly | Runs Isolation Forest + Gradient Boosting + SHAP on claim features |
| Explain | Generates human-readable SHAP explanation |
| Synth | Merges all agent outputs into one coherent response |

---

## Model Scores

### 1. Isolation Forest — Unsupervised Anomaly Detection

Trained on **220,663 rows** adapted from the IEEE-CIS Fraud Detection dataset (mapped to commute-claim domain). Contamination set from real fraud rate (9.36%).

| Metric | Score |
|---|---|
| ROC-AUC | **0.9541** |
| F1 Score | **0.6933** |
| Precision | 0.6933 |
| Recall | 0.6933 |
| Accuracy | **94.26%** |
| Dataset | IEEE-CIS adapted (220k rows, 9.4% anomaly) |

> Isolation Forest is **unsupervised** — it never sees labels during training. F1 of 0.69 on a real-world fraud distribution is strong for a label-free model.

---

### 2. Gradient Boosting Classifier — Supervised Anomaly Detection

Same 220k dataset, 75/25 train-test split, 5-fold cross-validation.

| Metric | Train | Test (held-out) |
|---|---|---|
| ROC-AUC | 0.9998 | **0.9997** |
| PR-AUC | 0.9987 | **0.9976** |
| F1 Score | 0.9853 | **0.9793** |
| Precision | 0.9902 | **0.9836** |
| Recall | 0.9805 | **0.9750** |
| Accuracy | 99.73% | **99.61%** |

**Cross-Validation (5-fold)**

| Metric | Mean | Std |
|---|---|---|
| ROC-AUC | 0.9997 | ±0.0001 |
| F1 | 0.9806 | ±0.0015 |

**Overfitting check:** Train-Test ROC-AUC gap = **0.0001** (no overfitting)

Features: `distance_km`, `delay_minutes`, `route_avg_delay_min`, `day_of_week`, `hour_of_day`, `claim_frequency_30d`, `delay_ratio`

---

### 3. Sentence Transformer — Fine-Tuned RAG Embeddings

Fine-tuned `all-mpnet-base-v2` on **1,247 (question, policy-chunk) pairs** generated from 665 chunks across 8 HR policy PDFs using MultipleNegativesRankingLoss.

| Metric | Before | After | Improvement |
|---|---|---|---|
| Recall@1 | 0.07 | **0.97** | **+1,286%** |
| Training pairs | — | 1,247 | — |
| Epochs | — | 5 | — |
| Training time | — | ~21 min | — |

> Recall@1 measures whether the most relevant policy chunk is retrieved as the top result. Fine-tuning improved retrieval from 7% to 97%.

---

## Dataset

**Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle)
- 590,540 original transactions → 220,663 after sampling & adaptation
- 3.5% original fraud rate → 9.4% commute-anomaly rate (includes route deviations, timing anomalies, frequency abuse)

**Adaptation** (`data/adapt_ieee_cis.py`):

| IEEE-CIS Field | Commute Field | Transformation |
|---|---|---|
| `isFraud` | `is_anomaly` | Direct mapping |
| `TransactionDT` | `day_of_week`, `hour_of_day` | Modulo arithmetic |
| `TransactionAmt` | `distance_km` | Scaled to 3–65 km (Bangalore routes) |
| `C1` | `claim_frequency_30d` | Scaled with overlapping fraud/normal ranges |
| Derived | `delay_minutes` | Route avg × 1.5–4× multiplier (fraud) |
| Derived | `delay_ratio` | `delay_minutes / route_avg_delay_min` |

**Data integrity checks performed:**
- No perfect separation in any feature (confirmed: `claim_frequency_30d` ranges overlap at 4-12 band)
- Delay ratio: mean 3.7× for anomalies (realistic vs. 8.5× before fix)
- Overfitting gap: 0.0001 ROC-AUC (train vs. test)

---

## RAG Pipeline

- **8 policy PDFs** → **665 chunks** (recursive character splitting, 500 chars / 50 overlap)
- **Embedder:** Fine-tuned `all-mpnet-base-v2` (→ `models/finetuned_embedder/`)
- **Vector store:** FAISS flat index
- **Retrieval:** Top-7 chunks per query
- **LLM Query Rewriting:** Follow-up questions are rewritten as standalone policy queries before retrieval (Conversational RAG pattern)
- **Conversation memory:** Last 6 messages sent with every request

**Policy documents indexed:**
- `COMMUTE_POLICY.pdf`
- `REIMBURSEMENT_AND_COMMERCIAL_POLICY.pdf`
- `EMERGENCY_PROCEDURES_SAFETY_PRIVACY_NDA.pdf`
- `DEFINITIONS_AND_INTERPRETATIONS.pdf`
- `EMPLOYEE FAQ.pdf` (Annexure E)
- `APPROVAL MATRIX.pdf` (Annexure F)
- `ROUTE MASTER.pdf` (Annexure G)
- `p463.pdf` (IRS Publication 463 — Travel, Gift, Car Expenses)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` |
| LLM | Llama 3.2 via Ollama (local, no API cost) |
| RAG | FAISS + fine-tuned Sentence Transformers |
| Anomaly Detection | Isolation Forest + Gradient Boosting |
| Explainability | SHAP TreeExplainer |
| API | FastAPI + Pydantic v2 |
| Frontend | Vanilla HTML/JS chat UI |
| Training | PyTorch + HuggingFace Transformers + `accelerate` |

---

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- `ollama pull llama3.2`

### Install
```bash
git clone https://github.com/<your-username>/multiagent-commute-ai.git
cd multiagent_commute_ai
pip install -r requirements.txt
```

### Run (VS Code — one click)
Open in VS Code → **Run & Debug** → select a config → **F5**

| Config | What it does |
|---|---|
| Run FastAPI Server | Starts the API + chat UI on `http://localhost:8000` |
| Retrain Anomaly Model | Re-trains Isolation Forest on adapted dataset |
| Train Supervised Model | Re-trains Gradient Boosting classifier |
| Rebuild RAG Index | Re-ingests PDFs and rebuilds FAISS index |
| Adapt IEEE-CIS Dataset | Runs the Kaggle → commute domain adapter |
| Fine-Tune Embeddings | Fine-tunes sentence-transformer on policy pairs |
| Run Tests | Runs pytest suite |

### Run (terminal)
```bash
# 1. Adapt the dataset (download ieee-fraud-detection from Kaggle first)
python data/adapt_ieee_cis.py --sample 200000

# 2. Train anomaly models
python -m ml.train_isolation_forest
python -m ml.train_supervised

# 3. Fine-tune embeddings
python -m ml.finetune_embeddings --epochs 5

# 4. Rebuild RAG index with fine-tuned embedder
python -m rag.rebuild_index

# 5. Start server
uvicorn main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## Project Structure

```
multiagent_commute_ai/
├── agents/
│   ├── intent_agent.py       # Query classification
│   ├── policy_agent.py       # RAG + LLM Query Rewriting
│   ├── anomaly_agent.py      # Isolation Forest + GBM inference
│   ├── explain_agent.py      # SHAP explanation
│   ├── synth_agent.py        # Response synthesis
│   └── state.py              # LangGraph shared state
├── data/
│   ├── adapt_ieee_cis.py     # IEEE-CIS → commute domain adapter
│   ├── route_master.csv      # 25 Bangalore commute routes
│   └── policies/             # 8 HR policy PDFs
├── graph/
│   └── workflow.py           # LangGraph StateGraph definition
├── ml/
│   ├── train_isolation_forest.py
│   ├── train_supervised.py
│   ├── finetune_embeddings.py
│   └── inference.py
├── models/
│   ├── isolation_forest.pkl
│   ├── supervised_model.pkl
│   ├── finetuned_embedder/   # Fine-tuned sentence-transformer
│   ├── validation_metrics.json
│   ├── supervised_metrics.json
│   └── embedding_finetune_log.json
├── rag/
│   ├── ingestion.py          # PDF chunking + FAISS indexing
│   └── retriever.py          # Similarity search
├── schemas/
│   └── api_schemas.py
├── static/
│   └── chat.html             # Chat UI
├── .vscode/
│   ├── launch.json           # F5 run configs
│   └── tasks.json
├── main.py                   # FastAPI entry point
└── requirements.txt
```

---

## Key Design Decisions

- **Local LLM only** — Llama 3.2 via Ollama, zero external API cost and full data privacy
- **LLM Query Rewriting** — Conversational RAG: follow-up questions are rewritten as complete standalone queries before FAISS retrieval, solving the context window problem
- **Unsupervised + Supervised ensemble** — Isolation Forest (no labels) + Gradient Boosting (with IEEE-CIS labels) cover both cold-start and labeled scenarios
- **SHAP explainability** — every anomaly flag comes with feature-level explanations for HR auditors
- **Fine-tuned embeddings** — domain-specific Q&A pairs improve policy retrieval from 7% → 97% Recall@1

---

## License

MIT
