"""Claude as the conversation brain: streams reply text and runs the clinic tools."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Callable

import anthropic

from .clinic import SIDE_EFFECT_TOOLS, TOOLS, Clinic

# Phrase boundaries where it is safe to hand text to TTS without hurting prosody.
_SENTENCE_END = re.compile(r"[.!?।]+[\"')\]]*\s")
_CLAUSE_END = re.compile(r"[,;:]\s")
# "Dr. Sharma" must not end a phrase.
_ABBREV = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|St|No|Sr|Jr)\.$", re.IGNORECASE)


class PhraseChunker:
    """Turns a stream of text deltas into speakable phrases.

    The first phrase is cut early (at a comma once it has a few words) so TTS can
    start sooner; after that we cut only at sentence ends for smoother prosody.
    """

    def __init__(self, first_min_chars: int = 12, max_chars: int = 160):
        self.buf = ""
        self.first = True
        self.first_min_chars = first_min_chars
        self.max_chars = max_chars

    def push(self, text: str) -> list[str]:
        self.buf += text
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            phrase, self.buf = self.buf[:cut], self.buf[cut:]
            if phrase.strip():
                out.append(phrase)
                self.first = False
        return out

    def _find_cut(self) -> int | None:
        for m in _SENTENCE_END.finditer(self.buf):
            if not _ABBREV.search(self.buf[: m.end()].rstrip()):
                return m.end()
        if self.first:
            for m in _CLAUSE_END.finditer(self.buf):
                if m.end() >= self.first_min_chars:
                    return m.end()
        if len(self.buf) >= self.max_chars:
            space = self.buf.rfind(" ", 0, self.max_chars)
            return space + 1 if space > 0 else self.max_chars
        return None

    def flush(self) -> str:
        rest, self.buf = self.buf, ""
        return rest


class LLMTurn:
    """One bot reply: may take several model rounds when tools are used.

    Nothing is written to the shared history until the reply finishes; the new
    messages are left in ``new_messages`` for the caller to commit. That way a
    cancelled (barge-in) or discarded (speculative miss) reply leaves no trace.
    """

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        *,
        model: str,
        effort: str,
        max_tokens: int,
        system: str,
        clinic: Clinic,
        history: list[dict[str, Any]],
        user_text: str,
        on_event: Callable[..., None],
        confirmed: asyncio.Event,
        max_rounds: int = 4,
    ):
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.system = system
        self.clinic = clinic
        self.base = list(history)
        self.user_text = user_text
        self.on_event = on_event  # on_event(name, **data) -> metrics
        self.confirmed = confirmed
        self.max_rounds = max_rounds
        self.new_messages: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
        self.text_out = ""
        self.usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self.rounds = 0
        self.tool_calls: list[dict[str, Any]] = []
        self.booking: dict[str, Any] | None = None

    async def phrases(self) -> AsyncIterator[str]:
        chunker = PhraseChunker()
        first_token = True
        for _ in range(self.max_rounds):
            self.rounds += 1
            self.on_event("llm_round_start", round=self.rounds)
            async with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                messages=self.base + self.new_messages,
                tools=TOOLS,
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        if first_token:
                            first_token = False
                            self.on_event("llm_first_token")
                        self.text_out += event.delta.text
                        for phrase in chunker.push(event.delta.text):
                            yield phrase
                message = await stream.get_final_message()

            self._add_usage(message.usage)
            self.on_event(
                "llm_round_end",
                round=self.rounds,
                stop_reason=message.stop_reason,
                model=message.model,
                usage=message.usage.model_dump() if message.usage else None,
            )
            self.new_messages.append({"role": "assistant", "content": message.content})

            if message.stop_reason == "refusal":
                self.on_event("llm_refusal", details=str(getattr(message, "stop_details", None)))
                break
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if message.stop_reason != "tool_use" or not tool_uses:
                break

            results = []
            for block in tool_uses:
                results.append(await self._run_tool(block))
            self.new_messages.append({"role": "user", "content": results})

        rest = chunker.flush()
        if rest.strip():
            yield rest

    async def _run_tool(self, block: Any) -> dict[str, Any]:
        args = block.input if isinstance(block.input, dict) else {}
        waited_ms = 0.0
        if block.name in SIDE_EFFECT_TOOLS and not self.confirmed.is_set():
            t = time.monotonic()
            await self.confirmed.wait()
            waited_ms = round((time.monotonic() - t) * 1000, 1)
        t = time.monotonic()
        result = self.clinic.run_tool(block.name, args)
        ms = round((time.monotonic() - t) * 1000, 2)
        call = {"name": block.name, "args": args, "ok": bool(result.get("ok")), "ms": ms,
                "waited_for_confirm_ms": waited_ms}
        self.tool_calls.append(call)
        self.on_event("tool_call", **call, result=result)
        if block.name == "book_appointment" and result.get("ok"):
            self.booking = result
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": json.dumps(result, ensure_ascii=False),
            "is_error": not result.get("ok", False),
        }

    def _add_usage(self, usage: Any) -> None:
        if not usage:
            return
        for k in self.usage:
            self.usage[k] += getattr(usage, k, 0) or 0

    def interrupted_messages(self, spoken_text: str) -> list[dict[str, Any]]:
        """History to keep when the user cut the bot off: their turn plus what was said."""
        said = spoken_text.strip() or "…"
        return [
            {"role": "user", "content": self.user_text},
            {"role": "assistant", "content": said + " [interrupted by caller]"},
        ]
