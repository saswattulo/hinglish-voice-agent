"""Stand-ins for Cartesia and Claude so a whole call can run offline."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any


class FakeSTT:
    def __init__(self):
        self.q: asyncio.Queue = asyncio.Queue()
        self.audio = bytearray()

    async def connect(self):
        pass

    async def send_audio(self, pcm):
        self.audio += pcm

    async def events(self):
        while True:
            ev = await self.q.get()
            if ev is None:
                return
            yield ev

    async def close(self):
        self.q.put_nowait(None)


class FakeContext:
    def __init__(self, delay: float):
        self.q: asyncio.Queue = asyncio.Queue()
        self.texts: list[str] = []
        self.cancelled = False
        self.delay = delay

    async def send(self, text):
        self.texts.append(text)
        await asyncio.sleep(self.delay)
        self.q.put_nowait(b"\x01\x00" * 240)  # 10 ms of audio per phrase

    async def end(self):
        self.q.put_nowait(None)

    async def cancel(self):
        self.cancelled = True
        self.q.put_nowait(None)

    async def audio(self):
        while True:
            item = await self.q.get()
            if item is None:
                return
            yield item


class FakeTTS:
    def __init__(self, delay: float = 0.0):
        self.contexts: list[FakeContext] = []
        self.delay = delay

    async def connect(self):
        pass

    def new_context(self):
        ctx = FakeContext(self.delay)
        self.contexts.append(ctx)
        return ctx

    async def close(self):
        pass


def text_reply(text: str) -> dict[str, Any]:
    return {"text": text, "tool": None}


def tool_reply(name: str, args: dict[str, Any], text: str = "") -> dict[str, Any]:
    return {"text": text, "tool": (name, args)}


class _Stream:
    def __init__(self, script: dict[str, Any], token_delay: float):
        self.script = script
        self.token_delay = token_delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        for word in self.script["text"].split(" "):
            if not word:
                continue
            await asyncio.sleep(self.token_delay)
            yield NS(type="content_block_delta", delta=NS(type="text_delta", text=word + " "))

    async def get_final_message(self):
        content = []
        if self.script["text"]:
            content.append(NS(type="text", text=self.script["text"]))
        stop = "end_turn"
        if self.script["tool"]:
            name, args = self.script["tool"]
            content.append(NS(type="tool_use", id=f"tu_{name}", name=name, input=args))
            stop = "tool_use"
        usage = NS(input_tokens=100, output_tokens=20, cache_read_input_tokens=80,
                   cache_creation_input_tokens=0, model_dump=lambda: {"input_tokens": 100})
        return NS(content=content, stop_reason=stop, usage=usage, model="fake-model")


class FakeClaude:
    """Returns scripted replies in order; records every request."""

    def __init__(self, replies: list[dict[str, Any]], token_delay: float = 0.0):
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.token_delay = token_delay
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kwargs):
        self.requests.append(kwargs)
        script = self.replies.pop(0) if self.replies else text_reply("ठीक है।")
        return _Stream(script, self.token_delay)
