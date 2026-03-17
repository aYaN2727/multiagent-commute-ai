<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=3000&pause=1000&color=1A73E8&center=true&vCenter=true&width=700&lines=Policy-Aware+Multi-Agent+GenAI;Employee+Commute+%26+Travel+Support;RAG+%2B+Anomaly+Detection+%2B+SHAP;Built+with+LangGraph+%2B+Ollama" alt="Typing SVG" />

<br/>

[![Made by](https://img.shields.io/badge/Made%20by-Ayan-blue?style=for-the-badge&logo=github)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B6B?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black?style=for-the-badge)](https://ollama.com)

<br/>

> An intelligent HR support system — answers policy questions, detects anomalous claims, and explains every decision.

</div>

---

## ✨ Features

<div align="center">

| | Feature | Description |
|:---:|---|---|
| 💬 | **Chat UI** | Browser-based chat at `/chat` — no extra tools needed |
| 📄 | **RAG Policy Q&A** | Upload PDFs; system indexes and retrieves relevant sections |
| 🎯 | **Intent Classification** | Auto-routes: policy question / delay claim / both / out-of-scope |
| 🚨 | **Anomaly Detection** | Isolation Forest flags suspicious delay claims with **96% recall** |
| 🔍 | **SHAP Explanations** | Top factors explained in plain English to the employee |
| 🦙 | **Runs Locally** | Fully local with Ollama + Llama 3.2 — no API key needed |
| 🧪 | **Mock Mode** | Test the full pipeline without any LLM |

</div>

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│              Employee Query                      │
│         Chat UI  /  REST API                     │
└─────────────────────┬────────────────────────────┘
                      │
               ┌──────▼──────┐
               │Intent Agent │  ← LLM classifies the query
               └──────┬──────┘
                      │
      ┌───────────────┼──────────────────┐
      │               │                  │
 policy_query    delay_claim /       out_of_scope
      │              both                │
 ┌────▼────┐    ┌─────▼─────┐      ┌────▼────┐
 │ Policy  │    │  Policy   │      │ Synth   │
 │  Agent  │    │  Agent    │      │  Agent  │
 │  (RAG)  │    └─────┬─────┘      └────┬────┘
 └────┬────┘          │                  │
      │        ┌──────▼──────┐           │
      │        │  Anomaly    │           │
      │        │   Agent     │           │
      │        │(Isol.Forest)│           │
      │        └──────┬──────┘           │
      │               │                  │
      │        ┌──────▼──────┐           │
      │        │  Explain    │           │
      │        │   Agent     │           │
      │        │   (SHAP)    │           │
      │        └──────┬──────┘           │
      │               │                  │
      └───────┬────────┘                 │
              │◄─────────────────────────┘
       ┌──────▼──────┐
       │ Synth Agent │  ← Assembles final response
       └──────┬──────┘
              │
     ┌────────▼────────┐
     │  Chat UI / JSON  │
     └─────────────────┘
```

---

## 🛠️ Tech Stack

<div align="center">

[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-FF6B6B?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2-black?style=flat-square)](https://ollama.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-Compatible-412991?style=flat-square&logo=openai)](https://openai.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-blue?style=flat-square)](https://github.com/facebookresearch/faiss)
[![SentenceTransformers](https://img.shields.io/badge/sentence--transformers-Embeddings-orange?style=flat-square)](https://sbert.net)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF%20Parsing-red?style=flat-square)](https://pymupdf.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen?style=flat-square)](https://shap.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)

</div>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed → `ollama pull llama3.2`
- **OR** an OpenAI API key
- 4 GB RAM minimum

### Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd multiagent_commute_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env: set LLM_PROVIDER=ollama (or openai + your key)

# 4. Drop your policy PDFs into data/policies/

# 5. Generate training data
python data/generate_commute_records.py

# 6. Train anomaly detection
python -m ml.train_isolation_forest

# 7. Build the RAG index
python -m rag.ingestion

# 8. Start the server
python main.py
```

Open **http://localhost:8000/chat** 🎉

---

## 📁 Project Structure

<details>
<summary><b>📂 Click to expand</b></summary>

```
multiagent_commute_ai/
│
├── 🤖 agents/
│   ├── intent_agent.py       # LLM-based query classifier
│   ├── policy_agent.py       # RAG retrieval + grounded answer
│   ├── anomaly_agent.py      # Isolation Forest inference
│   ├── explain_agent.py      # SHAP values + plain-English narrative
│   ├── synth_agent.py        # Final response assembly
│   └── state.py              # LangGraph AgentState TypedDict
│
├── ⚙️  config/
│   └── settings.py           # Pydantic-settings configuration
│
├── 📊 data/
│   ├── generate_commute_records.py
│   └── policies/             # ← Drop your PDF files here
│
├── 🔀 graph/
│   └── workflow.py           # LangGraph StateGraph definition
│
├── 🧠 ml/
│   ├── train_isolation_forest.py
│   └── inference.py
│
├── 🔍 rag/
│   ├── ingestion.py          # PDF → chunks → FAISS index
│   └── retriever.py          # Semantic search at query time
│
├── 🌐 static/
│   └── chat.html             # Browser chat UI
│
├── 📐 schemas/
│   └── api_schemas.py        # Pydantic v2 request/response models
│
├── 🛠️  utils/
│   ├── llm_client.py         # Async OpenAI/Ollama/mock wrapper
│   └── logger.py             # JSON structured logger
│
├── main.py                   # FastAPI entry point
├── test_pipeline.py          # End-to-end test suite
├── requirements.txt
└── .env.example
```

</details>

---

## 🔌 API Reference

<details>
<summary><b>📡 Click to expand endpoints</b></summary>

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/chat` | Browser chat UI |
| `POST` | `/query` | Main multi-agent pipeline |
| `GET` | `/health` | System status + model load check |
| `GET` | `/docs` | Swagger interactive docs |

### Policy question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"employee_id": "EMP_001", "query": "Who is eligible for commute transport?"}'
```

### Delay claim with commute record

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "employee_id": "EMP_001",
    "query": "My bus was 3 hours late, can I claim?",
    "commute_record": {
      "route_id": "ROUTE_07", "distance_km": 14.5,
      "delay_minutes": 180, "route_avg_delay_min": 8,
      "day_of_week": 1, "hour_of_day": 9,
      "claim_frequency_30d": 8, "week_num": 12, "is_holiday": 0
    }
  }'
```

### Response

```json
{
  "employee_id": "EMP_001",
  "intent": "delay_claim",
  "final_response": "Your delay claim has been flagged for manual review...",
  "anomaly_flagged": true,
  "anomaly_probability": 0.94,
  "shap_explanation": [
    "claim_frequency_30d = 8.0 pushed score UP by 0.21",
    "delay_minutes = 180.0 pushed score UP by 0.18"
  ],
  "needs_escalation": true
}
```

</details>

---

## 📦 Adding New Policy Documents

```bash
# Drop PDF(s) into the policies folder
cp your_policy.pdf data/policies/

# Rebuild the index
python -m rag.ingestion

# Restart the server
python main.py
```

---

## 🔧 Troubleshooting

<details>
<summary><b>⚠️ Common errors and fixes</b></summary>

| Error | Fix |
|---|---|
| `No module named 'fitz'` | `pip install pymupdf` |
| `FAISS index not found` | `python -m rag.ingestion` |
| `isolation_forest.pkl not found` | `python -m ml.train_isolation_forest` |
| `ollama: command not found` | Download from [ollama.com](https://ollama.com) |
| `ConnectError` to Ollama | Run `ollama serve` then `ollama pull llama3.2` |
| Port 8000 in use | Set `API_PORT=8001` in `.env` |

</details>

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=16&duration=4000&pause=500&color=1A73E8&center=true&vCenter=true&width=400&lines=Made+with+%E2%9D%A4%EF%B8%8F+by+Ayan;Star+the+repo+if+you+found+it+useful!" alt="Footer" />

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>
