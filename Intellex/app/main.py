"""Intellex FastAPI server.

The web UI is served from app/static. API credentials are server-side only:
the browser never receives OPENROUTER_API_KEY or any other provider key.
"""

import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .chatbot import ChatBot

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="Intellex")
bot = ChatBot()

RATE_LIMIT = max(1, int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")))
REBUILD_COOLDOWN = max(1, int(os.getenv("REBUILD_COOLDOWN_SECONDS", "60")))
_requests = defaultdict(deque)
_last_rebuild = defaultdict(float)


class ChatRequest(BaseModel):
    message: str
    rebuild_index: bool = False


class ChatResponse(BaseModel):
    answer: str
    case: int | None = None
    source: str | None = None
    db_results: list = []
    web_results: list = []
    aerocalc: dict | None = None
    mode: str | None = None


def _client_id(request: Request) -> str:
    # Cloudflare sets this header for proxied requests. Otherwise use the
    # direct socket address. Do not trust arbitrary X-Forwarded-For values.
    cloudflare_ip = request.headers.get("cf-connecting-ip")
    if cloudflare_ip:
        return cloudflare_ip.strip()
    return request.client.host if request.client else "unknown"


def _allowed(client_id: str) -> bool:
    now = time.monotonic()
    q = _requests[client_id]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return False
    q.append(now)
    return True


@app.on_event("startup")
def _startup():
    bot.ensure_index()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    return JSONResponse({"status": "ok", **bot.stats()})


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    client = _client_id(request)
    if not _allowed(client):
        return JSONResponse(
            {"error": "Too many requests. Please try again shortly."},
            status_code=429,
        )
    result = bot.answer(req.message, rebuild_index=req.rebuild_index)
    return ChatResponse(**result)


@app.post("/api/rebuild")
def rebuild(request: Request):
    client = _client_id(request)
    now = time.monotonic()
    if now - _last_rebuild[client] < REBUILD_COOLDOWN:
        return JSONResponse(
            {"error": "Please wait before rebuilding the index again."},
            status_code=429,
        )
    _last_rebuild[client] = now
    bot.ensure_index(force=True)
    return JSONResponse({"status": "rebuilt", **bot.stats()})


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
