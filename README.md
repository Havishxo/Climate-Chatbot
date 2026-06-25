# 🌍 ClimateBot — AI Climate Awareness Chatbot

> **Powered by IBM Granite · RAG · Agentic AI · Multimodal Inputs**

ClimateBot is a fully integrated AI assistant for climate science education and awareness. It combines IBM Granite LLMs with retrieval-augmented generation (RAG), agentic tool use, entity extraction, and multimodal document handling — all in a clean Gradio web interface with a FastAPI REST backend.

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Component Guide](#component-guide)
- [API Reference](#api-reference)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Extending the Bot](#extending-the-bot)

---

## ✨ Features

| Feature | Details |
|---|---|
| **Conversational AI** | IBM Granite 3.3-8B-Instruct via Watsonx.ai (Anthropic Claude as fallback) |
| **RAG Knowledge Base** | IPCC AR6, NOAA, NASA GISS, EPA, IEA, UNFCCC sources |
| **Live Climate Data** | Real-time CO₂ ppm + temperature anomaly from NOAA/NASA APIs |
| **Carbon Calculator** | Personal footprint estimate with reduction tips |
| **Document Ingestion** | PDF, CSV, TXT files auto-chunked and indexed |
| **Image Analysis** | Satellite images and climate charts described via vision model |
| **Entity Extraction** | Structured JSON: gases, locations, temperatures, orgs, policies |
| **Summarisation** | Map-reduce strategy for long IPCC-style reports |
| **Conversation Memory** | Rolling memory with auto-summarisation of old turns |
| **REST API** | Full FastAPI backend with Swagger docs |
| **Web UI** | Gradio 5 — chat + calculator + analysis + live data tabs |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│              Gradio Web UI  /  FastAPI REST                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      Agent Orchestrator                      │
│              Intent Classification → Tool Routing            │
└───┬─────────┬──────────────┬──────────────┬─────────────────┘
    │         │              │              │
    ▼         ▼              ▼              ▼
  RAG      Live Data    Carbon Calc    Entity/Summary
  Chain    (NOAA API)   (Calculator)   (NLP Pipeline)
    │
    ▼
┌────────────────────────────────┐
│     Vector Store (ChromaDB)    │
│  IPCC · NOAA · EPA · NASA      │
│  + User-uploaded documents     │
└────────────────────────────────┘
    │
    ▼
┌────────────────────────────────┐
│     IBM Granite 3.3-8B-Instruct│
│     (Watsonx.ai / local)       │
│     Fallback: Claude Sonnet    │
└────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd climate_chatbot
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```env
# IBM Watsonx (for IBM Granite — preferred)
IBM_API_KEY=your_ibm_api_key
IBM_PROJECT_ID=your_project_id
IBM_URL=https://us-south.ml.cloud.ibm.com

# Anthropic (fallback / development)
ANTHROPIC_API_KEY=your_anthropic_key
```

> **No API keys?** The app runs in demo mode with a MockLLM that returns sample climate answers. All other features (RAG, calculator, entity extraction) still work.

### 3. Launch

```bash
# Web UI (Gradio) — http://localhost:7860
python main.py

# REST API (FastAPI) — http://localhost:8000
python main.py --api

# Terminal demo (no browser needed)
python main.py --demo
```

---

## ⚙️ Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `IBM_API_KEY` | — | IBM Watsonx API key |
| `IBM_PROJECT_ID` | — | Watsonx project ID |
| `GRANITE_CHAT_MODEL` | `ibm/granite-3-3-8b-instruct` | Chat model ID |
| `GRANITE_EMBEDDING_MODEL` | `ibm/granite-embedding-107m-multilingual` | Embedding model |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (fallback) |
| `CHUNK_SIZE` | `512` | RAG chunk size in tokens |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `MAX_RETRIEVED_CHUNKS` | `5` | Top-k chunks retrieved per query |
| `CHROMA_DB_PATH` | `./data/chroma_db` | Persistent vector store path |

---

## 🧩 Component Guide

### 1. Prompt Engineering (`app/prompts.py`)

All prompt templates are centralised here:

```python
from app.prompts import build_rag_prompt

# Zero-shot RAG prompt
prompt = build_rag_prompt(context, question)

# Chain-of-Thought
prompt = build_rag_prompt(context, question, use_cot=True)

# Few-shot (pre-seeded Q&A examples)
prompt = build_rag_prompt(context, question, use_few_shot=True)
```

Techniques implemented:
- **System prompt with constraints** — cites sources, refuses fabrication
- **Chain-of-Thought (CoT)** — 4-step reasoning before answering
- **Few-shot examples** — 3 calibrated Q&A pairs
- **Dynamic context injection** — retrieved chunks under CONTEXT block
- **ReAct format** — for agentic Thought → Action → Observation loops

---

### 2. IBM Granite (`app/llm_provider.py`)

```python
from app.llm_provider import get_llm, generate

llm = get_llm()        # auto-selects IBM → Anthropic → Mock
answer = generate("What is climate change?")
```

Priority chain:
1. **IBM Granite 3.3-8B-Instruct** via Watsonx.ai (if credentials set)
2. **Anthropic Claude Sonnet** (if `ANTHROPIC_API_KEY` set)
3. **MockLLM** — keyword-aware demo responses (offline)

---

### 3. RAG (`app/rag.py`)

```python
from app.rag import rag_query, ingest_pdf, ingest_text

# Query with retrieval
result = rag_query("How fast is Arctic warming?")
print(result["answer"])    # cited answer
print(result["sources"])   # list of source documents

# Ingest your own PDF
chunks = ingest_pdf("/path/to/report.pdf")

# Ingest raw text
ingest_text("New climate data...", source="MyReport 2024")
```

Pipeline: user query → embed → Chroma similarity search → rerank → inject → Granite → answer

---

### 4. Agentic AI (`app/agent.py`, `app/tools.py`)

```python
from app.agent import agent_respond, ConversationMemory

memory = ConversationMemory()
result = agent_respond("What is the current CO2 level?", memory)
# → Automatically calls get_live_co2() tool and cites NOAA
```

Available tools:
- `get_live_co2()` — NOAA real-time CO₂ ppm
- `get_temperature_anomaly()` — NASA GISS temperature anomaly
- `calculate_carbon_footprint(...)` — personal CO₂ estimate
- `search_knowledge_base(query)` — RAG retrieval as tool
- `summarise_text(text)` — on-demand summarisation

---

### 5. Entity Extraction & Summarisation (`app/entity_extraction.py`)

```python
from app.entity_extraction import extract_entities, summarise_document

# Extract structured entities
entities = extract_entities("The IPCC reports 1.5°C warming with CO2 at 421 ppm...")
# → {"greenhouse_gases": ["CO2"], "temperatures": ["1.5°C"], "organizations": ["IPCC"], ...}

# Summarise a long document (map-reduce for > 1000 words)
result = summarise_document(long_text, max_summary_words=200, style="accessible")
print(result["summary"])
```

---

### 6. Multimodal Inputs (`app/multimodal.py`)

```python
from app.multimodal import process_upload

result = process_upload("/path/to/report.pdf")
result = process_upload("/path/to/satellite.png")
result = process_upload("/path/to/climate_data.csv")
# All return: {type, text_extracted, chunks_added, entities, description}
```

---

## 🌐 API Reference

Start the server: `python main.py --api`

Full docs: http://localhost:8000/docs

### Key endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/chat` | Main Q&A with RAG + agents |
| `POST` | `/upload` | Ingest PDF / image / CSV |
| `GET` | `/live/co2` | Real-time CO₂ data |
| `GET` | `/live/temperature` | Temperature anomaly |
| `POST` | `/footprint` | Carbon footprint calc |
| `POST` | `/extract` | Entity extraction |
| `POST` | `/summarise` | Document summarisation |
| `GET` | `/stats` | System statistics |
| `GET` | `/health` | Health check |
| `DELETE` | `/session/{id}` | Clear chat memory |

### Example chat request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the current CO2 level?", "session_id": "my-session"}'
```

---

## 🧪 Running Tests

```bash
# Run all tests
python tests/test_all.py

# With pytest (verbose)
pip install pytest
pytest tests/ -v

# Specific test class
pytest tests/test_all.py::TestTools -v
```

Test coverage:
- ✅ Config loading
- ✅ LLM provider (Mock, IBM, Anthropic)
- ✅ TF-IDF embeddings
- ✅ All agent tools (with mocked APIs)
- ✅ Entity extraction (LLM + regex fallback)
- ✅ Agent intent classification + memory
- ✅ RAG ingestion + retrieval
- ✅ Multimodal file processing (PDF, CSV, TXT, image)
- ✅ All FastAPI endpoints

---

## 📁 Project Structure

```
climate_chatbot/
├── main.py                   # Entry point (UI / API / demo)
├── api.py                    # FastAPI REST server
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
│
├── app/
│   ├── __init__.py
│   ├── config.py             # All settings from .env
│   ├── llm_provider.py       # IBM Granite + Anthropic + Mock
│   ├── embeddings.py         # IBM Granite / HuggingFace / TF-IDF
│   ├── prompts.py            # All prompt templates
│   ├── rag.py                # RAG pipeline + Chroma vector store
│   ├── tools.py              # Agentic tools (CO2, temp, footprint)
│   ├── agent.py              # ReAct orchestrator + memory
│   ├── entity_extraction.py  # NER + summarisation
│   ├── multimodal.py         # PDF / image / CSV processing
│   └── ui.py                 # Gradio web interface
│
├── data/
│   ├── knowledge_base/       # Place raw documents here for ingestion
│   └── chroma_db/            # Persistent vector store (auto-created)
│
├── tests/
│   └── test_all.py           # Full test suite (90+ tests)
│
├── static/                   # Static assets for UI
└── models/                   # Local model cache (if used)
```

---

## 🔧 Extending the Bot

### Add a new tool

```python
# In app/tools.py
def my_new_climate_tool(param: str) -> dict:
    """Fetch data from a new climate API."""
    # ... implementation
    return {"data": ..., "source": "My API"}

# Register it
TOOL_REGISTRY["my_new_climate_tool"] = {
    "fn": my_new_climate_tool,
    "description": "Fetches X from Y API.",
    "args": ["param"],
}
```

### Add new knowledge sources

```python
# Ingest a directory of PDFs on startup
from app.rag import ingest_pdf
import glob

for pdf in glob.glob("data/knowledge_base/*.pdf"):
    ingest_pdf(pdf)
```

### Switch to a different vector store

Replace `Chroma` with `Milvus`, `Pinecone`, or `FAISS` in `app/rag.py`:

```python
# Pinecone example
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore(index_name="climate-bot", embedding=embeddings)
```

---

## 📚 Knowledge Sources

| Source | Coverage |
|---|---|
| **IPCC AR6 (2021-2022)** | Physical science, impacts, mitigation |
| **NOAA Global Monitoring** | CO₂, methane, temperature records |
| **NASA GISS** | Surface temperature analysis |
| **EPA GHG Inventory** | US emissions data |
| **IEA World Energy Outlook** | Renewable energy, fossil fuels |
| **UNFCCC** | Paris Agreement, NDCs |
| **IUCN** | Nature-based solutions |
| **Global CCS Institute** | Carbon capture status |

---

## 📄 License

MIT License — free for research, education, and commercial use.

---

*Built with ❤️ for climate awareness. The planet needs all the intelligence it can get.*
