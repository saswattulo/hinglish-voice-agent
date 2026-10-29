"""FastAPI app: serves the call page, the metrics page and the /ws call socket."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .cartesia import CartesiaSTT, CartesiaTTS
from .config import settings
from .metrics import aggregate, load_summaries
from .session import CallSession, parse_client_text

log = logging.getLogger("voicebot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

STATIC = Path(__file__).resolve().parent.parent / "static"
app = FastAPI(title="Hinglish clinic voice bot")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/dashboard")
async def dashboard() -> FileResponse:
    return FileResponse(STATIC / "dashboard.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": not settings.missing_keys(),
        "missing_keys": settings.missing_keys(),
        "stt_model": settings.stt_model,
        "tts_model": settings.tts_model,
        "llm_model": settings.llm_model,
        "speculative_llm": settings.speculative_llm,
    }


@app.get("/api/calls")
async def list_calls(limit: int = 50) -> dict:
    summaries = load_summaries(settings.metrics_dir, limit=limit)
    calls = [
        {
            "call_id": s["call_id"],
            "started_at": s["started_at"],
            "duration_ms": s["duration_ms"],
            "end_reason": s["end_reason"],
            "turns": len(s["turns"]),
            "booked": bool(s.get("outcome", {}).get("booking")),
            "totals": s["totals"],
            "derived_ms": s["derived_ms"],
        }
        for s in summaries
    ]
    return {"aggregate": aggregate(summaries), "calls": calls}


@app.get("/api/calls/{call_id}")
async def get_call(call_id: str) -> JSONResponse:
    if not re.fullmatch(r"[0-9A-Za-z-]+", call_id):
        raise HTTPException(400, "bad call id")
    path = settings.metrics_dir / "calls" / f"{call_id}.summary.json"
    if not path.exists():
        raise HTTPException(404, "call not found (still in progress?)")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.websocket("/ws")
async def call_socket(ws: WebSocket) -> None:
    await ws.accept()
    missing = settings.missing_keys()
    if missing:
        await ws.send_json({"type": "error", "where": "config", "message": f"Missing {', '.join(missing)} in .env"})
        await ws.close()
        return

    session = CallSession(
        settings=settings,
        send_json=ws.send_json,
        send_bytes=ws.send_bytes,
        stt=CartesiaSTT(
            settings.cartesia_api_key,
            model=settings.stt_model,
            sample_rate=settings.stt_sample_rate,
            turn_end_timeout_ms=settings.stt_turn_end_timeout_ms,
            keyterms=settings.stt_keyterms,
        ),
        tts=CartesiaTTS(
            settings.cartesia_api_key,
            model=settings.tts_model,
            voice_id=settings.tts_voice_id,
            language=settings.tts_language,
            sample_rate=settings.tts_sample_rate,
            speed=settings.tts_speed,
            emotion=settings.tts_emotion,
            max_buffer_delay_ms=settings.tts_max_buffer_delay_ms,
        ),
        llm_client=anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key),
    )
    log.info("call %s started", session.metrics.call_id)
    reason = "client_disconnected"
    try:
        await session.start()
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await session.on_mic_audio(msg["bytes"])
            elif msg.get("text"):
                data = parse_client_text(msg["text"])
                if data.get("type") == "hangup":
                    reason = "hangup"
                    break
                await session.on_client_message(data)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # keep the metrics even when the call dies
        log.exception("call %s failed", session.metrics.call_id)
        session.metrics.error("server", repr(e))
        reason = "server_error"
    finally:
        summary = await session.close(reason)
        log.info("call %s ended (%s): %d turns", summary["call_id"], reason, len(summary["turns"]))
        try:
            await ws.close()
        except Exception:
            pass
