# ARIVANTI — BIS Standards Recommendation Engine

An AI-powered Retrieval-Augmented Generation (RAG) system that instantly recommends relevant Bureau of Indian Standards (BIS) for building materials. Built for the BIS Hackathon 2026.

## 🎯 What It Does

MSEs (Micro and Small Enterprises) spend weeks finding which BIS standards apply to their products. ARIVANTI turns product descriptions into accurate standard recommendations **in under 0.05 seconds**.

## 🏆 Performance Metrics

| Metric | Score | Target |
|---|---|---|
| Hit Rate @3 | **100.00%** | > 80% |
| MRR @5 | **0.8833** | > 0.7 |
| Avg Latency | **0.02 sec** | < 5 sec |

## 🏗️ Architecture

```
User Query → Intent Graph → FAISS Vector Search → Cross-Reference Mining → LLM Summary
```

1. **Intent Detection** (`intent_graph.py`): Keyword-based section classifier narrows search scope across 23 building material categories.
2. **FAISS Vector Search** (`query_engine.py`): Embeds query using `all-MiniLM-L6-v2` and searches pre-filtered chunk vectors for top matches.
3. **Cross-Reference Extraction**: RegEx mining of retrieved chunks to discover related IS codes mentioned in document text.
4. **LLM Summarization** (`llm_client.py`): Sends retrieved chunks to a local LLM (Ollama/LM Studio) for human-friendly answers.

## 📦 Tech Stack

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dim, sentence-transformers)
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **LLM**: Ollama with Llama 3 (pluggable — supports LM Studio, any OpenAI-compatible API)
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JS with glassmorphic design
- **Dataset**: SP 21:2005 — 450 chunked standards across 23 sections

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) (optional, for AI summaries)

### Setup
```bash
# Clone the repository
git clone https://github.com/Anshulpj12/ARIVA.git
cd ARIVA

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the App
```bash
# Windows (one-click):
start.bat

# Or manually:
python backend/ingest.py    # Build FAISS index (first time only)
python backend/server.py    # Start server on http://localhost:8000
```

### ⚖️ Automated Evaluation (For Judges)

The system is fully compliant with the Hackathon Rulebook for automated scoring. You do not need to build the FAISS index or download any LLM models for the automated scoring—the required index is pre-compiled in the repo.

**Step 1: Run Inference on Hidden Dataset**
Use the mandatory `inference.py` script to process your hidden test set:
```bash
python inference.py --input path/to/hidden_private_dataset.json --output team_results.json
```
*This will instantly execute the intent detection and FAISS vector retrieval pipeline, saving the output in the strict required JSON format.*

**Step 2: Calculate Score**
Run the official evaluation script to verify our 100% Hit Rate and Sub-second Latency:
```bash
python eval_script.py --results team_results.json
```

## 📁 Project Structure

```
ARIVANTI/
├── backend/
│   ├── ingest.py           # Data ingestion & FAISS index builder
│   ├── intent_graph.py     # Graph-based intent detection (23 sections)
│   ├── query_engine.py     # Core RAG pipeline (embed → search → answer)
│   ├── llm_client.py       # Pluggable LLM client (Ollama/LM Studio/OpenAI)
│   ├── server.py           # FastAPI server with REST endpoints
│   ├── test_accuracy.py    # 20-query accuracy test suite
│   └── index_store/        # Pre-built FAISS index + metadata
├── frontend/
│   ├── index.html          # Main web interface
│   ├── app.js              # Frontend logic
│   └── style.css           # Glassmorphic dark theme
├── data/                   # Public test set evaluation results
├── inference.py            # Mandatory judge entry point
├── eval_script.py          # Organizer's evaluation script
├── config.json             # LLM/embedding model configuration
├── dataset.json            # BIS standards dataset (450 chunks)
├── requirements.txt        # Python dependencies
└── start.bat               # One-click Windows launcher
```

## 🔧 Configuration

Edit `config.json` to switch LLM providers without code changes:

```json
{
  "llm": {
    "active_provider": "ollama",
    "active_model": "llama3"
  }
}
```

Supported providers: Ollama, LM Studio, any OpenAI-compatible API.

## 👥 Team ARIVANTI
Built for the BIS Standards Recommendation Engine Hackathon 2026.
