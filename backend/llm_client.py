"""
Pluggable LLM Client — Supports Ollama, LM Studio, and any OpenAI-compatible API.
Reads active provider/model from config.json. Switchable without code changes.
"""
import json
import httpx
from pathlib import Path
from typing import Optional

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

def load_llm_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    provider_key = cfg["llm"]["active_provider"]
    model = cfg["llm"]["active_model"]
    provider = cfg["llm"]["available_providers"][provider_key]
    return provider_key, model, provider

SYSTEM_PROMPT = """You are a precise technical assistant specialized in Indian Standards for Building Materials (SP 21:2005, Bureau of Indian Standards).

RULES:
1. Answer ONLY from the provided context chunks. Never use outside knowledge.
2. Always cite the IS code (e.g., IS 269:1989) and section name in your answer.
3. If the context does not contain the answer, say "The provided standards do not contain information about this query."
4. For numerical values (strengths, percentages, dimensions), quote them exactly as stated.
5. Structure your answer clearly with headings or bullet points when appropriate.
6. If multiple IS standards are relevant, compare or list them clearly."""

def build_context_prompt(query: str, chunks: list) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"--- Source {i} ---\n"
            f"IS Code: {chunk['is_code']}\n"
            f"Title: {chunk['title']}\n"
            f"Section: {chunk['section']}\n"
            f"Page: {chunk['page_start']}\n"
            f"Content:\n{chunk['content'][:3000]}\n"
        )
    context_str = "\n".join(context_parts)
    return f"Based on the following Indian Standards documents, answer the user's question.\n\n{context_str}\n\nUser Question: {query}"

async def query_ollama(model: str, base_url: str, system: str, user_msg: str, timeout: float = 120.0) -> str:
    url = f"{base_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "No response from model.")

async def query_openai_compatible(model: str, base_url: str, api_path: str, system: str, user_msg: str, timeout: float = 120.0) -> str:
    url = f"{base_url}{api_path}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

async def generate_answer(query: str, chunks: list) -> str:
    """Main entry: send query + retrieved chunks to the configured LLM."""
    provider_key, model, provider = load_llm_config()
    user_msg = build_context_prompt(query, chunks)

    try:
        if provider_key == "ollama":
            return await query_ollama(model, provider["base_url"], SYSTEM_PROMPT, user_msg, timeout=300.0)
        else:
            return await query_openai_compatible(
                model, provider["base_url"], provider["api_path"], SYSTEM_PROMPT, user_msg, timeout=300.0
            )
    except httpx.ConnectError:
        return (
            f"❌ Could not connect to {provider['name']} at {provider['base_url']}.\n\n"
            f"Please ensure your LLM server is running:\n"
            f"• Ollama: Run `ollama serve` then `ollama pull {model}`\n"
            f"• LM Studio: Start the local server in LM Studio settings\n"
            f"• Other: Start your OpenAI-compatible server"
        )
    except httpx.TimeoutException:
        return f"❌ LLM error: Request timed out. The model is taking too long to load or generate. Try again or consider a lighter model."
    except httpx.HTTPStatusError as e:
        return f"❌ LLM returned error {e.response.status_code}: {e.response.text[:500]}"
    except Exception as e:
        err_msg = str(e)
        if not err_msg:
            err_msg = repr(e)
        return f"❌ LLM error: {err_msg}"
