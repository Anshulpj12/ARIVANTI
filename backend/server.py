"""
FastAPI Server — SP 21:2005 RAG System
Endpoints: POST /query, GET /sections, GET /stats, GET /config, POST /config
"""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

import query_engine
from intent_graph import SECTION_NAMES

app = FastAPI(title="SP 21:2005 RAG System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


class QueryRequest(BaseModel):
    question: str
    section_filter: Optional[int] = None


class ConfigUpdate(BaseModel):
    embedding_model: Optional[str] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None


@app.on_event("startup")
async def startup():
    query_engine.initialize()


@app.post("/query")
async def handle_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    result = await query_engine.process_query(req.question)
    return result


@app.get("/sections")
async def get_sections():
    return {"sections": [{"id": k, "name": v} for k, v in sorted(SECTION_NAMES.items())]}


@app.get("/stats")
async def get_stats():
    return query_engine.get_stats()


@app.get("/config")
async def get_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/config")
async def update_config(update: ConfigUpdate):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    changed = []
    if update.llm_provider and update.llm_provider in cfg["llm"]["available_providers"]:
        cfg["llm"]["active_provider"] = update.llm_provider
        changed.append(f"LLM provider → {update.llm_provider}")
    if update.llm_model:
        cfg["llm"]["active_model"] = update.llm_model
        changed.append(f"LLM model → {update.llm_model}")
    if update.embedding_model and update.embedding_model in cfg["embedding"]["available_models"]:
        cfg["embedding"]["active_model"] = update.embedding_model
        changed.append(f"Embedding → {update.embedding_model} (re-run ingest.py --rebuild)")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return {"status": "ok", "changes": changed}


# Serve frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


if __name__ == "__main__":
    import uvicorn
    with open(CONFIG_PATH, "r") as f:
        srv = json.load(f).get("server", {})
    uvicorn.run(app, host=srv.get("host", "0.0.0.0"), port=srv.get("port", 8000))
