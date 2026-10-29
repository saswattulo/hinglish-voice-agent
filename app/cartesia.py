"""Thin clients for Cartesia's streaming speech-to-text and text-to-speech WebSockets.

Raw WebSockets rather than the SDK, so every byte in and out can be timed.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from typing import Any, AsyncIterator
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect

from .config import CARTESIA_VERSION

STT_URL = "wss://api.cartesia.ai/stt/turns/websocket"
TTS_URL = "wss://api.cartesia.ai/tts/websocket"


class CartesiaSTT:
    """Ink speech-to-text with server-side turn detection.

    Emits ``turn.start``, ``turn.update``, ``turn.eager_end``, ``turn.resume``,
    ``turn.end`` and ``error`` events as dicts.
    """

    def __init__(self, api_key: str, *, model: str, sample_rate: int,
                 turn_end_timeout_ms: int, keyterms: tuple[str, ...] = ()):
        params: list[tuple[str, str]] = [
            ("model", model),
            ("encoding", "pcm_s16le"),
            ("sample_rate", str(sample_rate)),
            ("cartesia_version", CARTESIA_VERSION),
            ("turn_end_timeout_ms", str(turn_end_timeout_ms)),
        ]
        params += [("keyterm", k) for k in keyterms]
        self.url = f"{STT_URL}?{urlencode(params)}"
        self.api_key = api_key
        self.ws: ClientConnection | None = None

    async def connect(self) -> None:
        self.ws = await connect(
            self.url,
            additional_headers={"X-API-Key": self.api_key, "Cartesia-Version": CARTESIA_VERSION},
            max_size=None,
        )

    async def send_audio(self, pcm: bytes) -> None:
        assert self.ws
        await self.ws.send(pcm)

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        assert self.ws
        async for raw in self.ws:
            if isinstance(raw, bytes):
                continue
            yield json.loads(raw)

    async def close(self) -> None:
        if not self.ws:
            return
        try:
            await self.ws.send(json.dumps({"type": "close"}))
        except Exception:
            pass
        await self.ws.close()


class TTSContext:
    """One bot utterance. Text goes in piece by piece; audio comes out as it is made."""

    def __init__(self, tts: "CartesiaTTS", context_id: str):
        self.tts = tts
        self.id = context_id
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.closed_input = False

    async def send(self, text: str) -> None:
        await self.tts._send(self.id, text, cont=True)

    async def end(self) -> None:
        if not self.closed_input:
            self.closed_input = True
            await self.tts._send(self.id, "", cont=False)

    async def cancel(self) -> None:
        self.closed_input = True
        await self.tts._cancel(self.id)
        self.queue.put_nowait({"type": "done", "cancelled": True})

    async def audio(self) -> AsyncIterator[bytes]:
        """Yields raw PCM until the context is done. Raises on a TTS error."""
        while True:
            msg = await self.queue.get()
            kind = msg.get("type")
            if kind == "chunk" and msg.get("data"):
                yield base64.b64decode(msg["data"])
            elif kind == "error":
                raise RuntimeError(f"TTS error: {msg.get('title')}: {msg.get('message')}")
            if kind == "done" or (kind == "chunk" and msg.get("done")):
                return


class CartesiaTTS:
    """Sonic text-to-speech over one WebSocket per call, one context per utterance."""

    def __init__(self, api_key: str, *, model: str, voice_id: str, language: str,
                 sample_rate: int, speed: float, emotion: str, max_buffer_delay_ms: int):
        self.api_key = api_key
        self.base = {
            "model_id": model,
            "voice": {"id": voice_id},
            "language": language,
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": sample_rate},
            "generation_config": {"speed": speed, "emotion": emotion},
            "max_buffer_delay_ms": max_buffer_delay_ms,
        }
        self.ws: ClientConnection | None = None
        self.contexts: dict[str, TTSContext] = {}
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        self.ws = await connect(
            f"{TTS_URL}?{urlencode({'cartesia_version': CARTESIA_VERSION})}",
            additional_headers={"X-API-Key": self.api_key, "Cartesia-Version": CARTESIA_VERSION},
            max_size=None,
        )
        self._reader = asyncio.create_task(self._read())

    def new_context(self) -> TTSContext:
        ctx = TTSContext(self, uuid.uuid4().hex)
        self.contexts[ctx.id] = ctx
        return ctx

    async def _send(self, context_id: str, text: str, cont: bool) -> None:
        assert self.ws
        payload = {**self.base, "context_id": context_id, "transcript": text, "continue": cont}
        async with self._lock:
            await self.ws.send(json.dumps(payload, ensure_ascii=False))

    async def _cancel(self, context_id: str) -> None:
        assert self.ws
        async with self._lock:
            await self.ws.send(json.dumps({"context_id": context_id, "cancel": True}))

    async def _read(self) -> None:
        assert self.ws
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                ctx = self.contexts.get(msg.get("context_id", ""))
                if ctx:
                    ctx.queue.put_nowait(msg)
                    if msg.get("type") == "done" or msg.get("done"):
                        self.contexts.pop(ctx.id, None)
        except Exception as e:  # connection dropped: unblock anyone waiting on audio
            for ctx in list(self.contexts.values()):
                ctx.queue.put_nowait({"type": "error", "title": "connection", "message": str(e)})

    async def close(self) -> None:
        if self._reader:
            self._reader.cancel()
        if self.ws:
            await self.ws.close()
