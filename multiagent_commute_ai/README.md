# Policy-Aware Multi-Agent GenAI — Employee Commute & Travel Support

> **Made by Ayan**

An intelligent HR support system that answers employee questions about commute and travel policies using RAG (FAISS + sentence-transformers), detects anomalous delay claims using Isolation Forest, and explains every decision with SHAP values — all orchestrated by LangGraph agents, exposed via FastAPI, and usable through a built-in chat UI.

---

## Features

- **Chat UI** — Clean browser-based one-to-one chat at `/chat`
- **RAG Policy Q&A** — Upload your own PDF policy documents; the system indexes and retrieves relevant sections
- **Intent Classification** — Automatically routes queries (policy question / delay claim / both / out-of-scope)
- **Anomaly Detection** — Isolation Forest flags suspicious commute delay claims (96% recall)
- **SHAP Explanations** — Top contributing factors explained in plain English to the employee
- **Ollama Support** — Run fully local with Llama 3.2 (no OpenAI key needed)
- **Mock Mode** — Test the full pipeline without any LLM

---

## Architecture

```
Employee Query (Chat UI / REST API)
           |
     [Intent Agent]  -- LLM classifies intent
           |
    -------+--------+-----------+
    |               |           |
[Policy Agent]  [Policy +   [Out of scope]
 RAG retrieval   Anomaly]        |
    |               |            |
    |        [Anomaly Agent]     |
    |        Isolation Forest    |
    |               |            |
    |        [Explain Agent]     |
    |          SHAP values       |
    |               |            |
    +-------[Synth Agent]--------+
              Final reply
                  |
           [Chat UI / JSON]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Ollama (Llama 3.2) / OpenAI |
| RAG | FAISS + sentence-transformers |
| PDF parsing | PyMuPDF |
| Anomaly detection | scikit-learn Isolation Forest |
| Explainability | SHAP TreeExplainer |
| API | FastAPI + uvicorn |
| Chat UI | Vanilla HTML/CSS/JS |

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running (`ollama pull llama3.2`)
- **OR** an OpenAI API key
- 4 GB RAM minimum

---

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd multiagent_commute_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env — set LLM_PROVIDER=ollama (or openai + OPENAI_API_KEY)

# 4. Add your policy PDFs
# Drop your PDF files into:  data/policies/

# 5. Generate training data
python data/generate_commute_records.py

# 6. Train anomaly detection models
python -m ml.train_isolation_forest

# 7. Build the policy index (RAG)
python -m rag.ingestion

# 8. Start the server
python main.py
```

Open **http://localhost:8000/chat** to start chatting.

---

## Adding New Policy Documents

1. Drop PDF files into `data/policies/`
2. Rebuild the index:
   ```bash
   python -m rag.ingestion
   ```
3. Restart the server.

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/chat` | GET | Browser chat UI |
| `/query` | POST | Main multi-agent pipeline |
| `/health` | GET | System status |
| `/docs` | GET | Swagger UI |

### Example query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "employee_id": "EMP_001",
    "query": "Who is eligible for commute transport?"
  }'
```

---

## Project Structure

```
multiagent_commute_ai/
├── agents/          # LangGraph agent nodes
│   ├── intent_agent.py
│   ├── policy_agent.py
│   ├── anomaly_agent.py
│   ├── explain_agent.py
│   └── synth_agent.py
├── config/          # Settings (pydantic-settings)
├── data/            # Training data + policy PDFs
│   └── policies/    # <-- Drop your PDF files here
├── graph/           # LangGraph workflow definition
├── indexes/         # FAISS index (auto-generated)
├── ml/              # Isolation Forest training & inference
├── models/          # Saved model files (auto-generated)
├── rag/             # PDF ingestion, embedding, retrieval
├── schemas/         # Pydantic API schemas
├── static/          # Chat UI (chat.html)
├── utils/           # LLM client, logger
└── main.py          # FastAPI entry point
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'fitz'` | `pip install pymupdf` |
| `FAISS index not found` | Run `python -m rag.ingestion` |
| `Model not found: isolation_forest.pkl` | Run `python -m ml.train_isolation_forest` |
| `ollama: command not found` | Install from https://ollama.com |
| `ConnectError` to Ollama | Run `ollama serve` and `ollama pull llama3.2` |
| Port 8000 in use | Change `API_PORT` in `.env` |

---

Made by Ayan
