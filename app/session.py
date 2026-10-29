"""One browser call: mic audio -> Cartesia STT -> Claude -> Cartesia TTS -> speaker.

Turn-taking rules:
* STT's ``turn.end`` hands the user's words to Claude.
* With speculation on, Claude starts on ``turn.eager_end``. If the final transcript
  matches, its work is kept (a "hit"); otherwise it is thrown away. Booking never runs
  until the turn is confirmed over.
* A ``turn.start`` while the bot is talking is a barge-in: the bot stops, the browser
  drops buffered audio, and history records what the caller actually heard.

Audio frames sent to the browser are prefixed with the turn index (uint32 LE) so the
browser can drop frames from a turn it has already cleared.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import struct
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from .clinic import GREETING, Clinic, system_prompt
from .config import Settings
from .llm import LLMTurn
from .metrics import CallMetrics, TurnMetrics

APOLOGY = "माफ़ कीजिए, एक second, क्या आप दोबारा बोल सकते हैं?"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class Reply:
    turn: TurnMetrics
    transcript: str
    llm: LLMTurn | None = None
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    confirmed: asyncio.Event = field(default_factory=asyncio.Event)
    producer: asyncio.Task | None = None
    speaker: asyncio.Task | None = None
    spoken: str = ""
    committed: bool = False

    async def phrases(self) -> AsyncIterator[str]:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item

    def active(self) -> bool:
        return any(t and not t.done() for t in (self.producer, self.speaker))


class CallSession:
    def __init__(
        self,
        *,
        settings: Settings,
        send_json: Callable[[dict[str, Any]], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        stt: Any,
        tts: Any,
        llm_client: Any,
        clinic: Clinic | None = None,
    ):
        self.s = settings
        self._send_json = send_json
        self._send_bytes = send_bytes
        self.stt = stt
        self.tts = tts
        self.llm_client = llm_client
        self.clinic = clinic or Clinic()
        self.system = system_prompt(self.clinic.today) + f'\n\nYou already greeted the caller with: "{GREETING}"'
        self.history: list[dict[str, Any]] = []
        self.metrics = CallMetrics(
            settings.metrics_dir,
            config={
                "stt_model": settings.stt_model,
                "stt_turn_end_timeout_ms": settings.stt_turn_end_timeout_ms,
                "tts_model": settings.tts_model,
                "tts_voice_id": settings.tts_voice_id,
                "tts_language": settings.tts_language,
                "tts_speed": settings.tts_speed,
                "tts_emotion": settings.tts_emotion,
                "tts_max_buffer_delay_ms": settings.tts_max_buffer_delay_ms,
                "llm_model": settings.llm_model,
                "llm_effort": settings.llm_effort,
                "speculative_llm": settings.speculative_llm,
            },
            tts_sample_rate=settings.tts_sample_rate,
        )
        self.user_turn: TurnMetrics | None = None
        self.reply: Reply | None = None  # the reply being spoken (or about to be)
        self.spec: Reply | None = None  # speculative reply waiting for turn.end
        # Turn whose audio the browser is (probably) still playing, and when we expect it
        # to finish. The browser's playback_done clears it sooner; the estimate covers
        # browsers that never report back.
        self.client_playing_turn: int | None = None
        self._playback_end_ms = 0.0
        self._stt_task: asyncio.Task | None = None
        self._closed = False

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        await self.send({"type": "call_started", "call_id": self.metrics.call_id})
        self.metrics.mark_call("stt_connect_start")
        self.metrics.mark_call("tts_connect_start")

        async def connect_stt() -> None:
            await self.stt.connect()
            self.metrics.mark_call("stt_connected")

        async def connect_tts() -> None:
            await self.tts.connect()
            self.metrics.mark_call("tts_connected")

        await asyncio.gather(connect_stt(), connect_tts())
        self._stt_task = asyncio.create_task(self._read_stt())
        greeting = Reply(turn=self.metrics.new_turn(), transcript="")
        greeting.turn.bot_text = GREETING
        greeting.confirmed.set()
        greeting.committed = True
        greeting.queue.put_nowait(GREETING)
        greeting.queue.put_nowait(None)
        self._start_speaking(greeting)
        await self.send({"type": "bot_text", "turn": greeting.turn.index, "text": GREETING})

    async def close(self, reason: str) -> dict[str, Any]:
        if self._closed:
            return self.metrics.summary()
        self._closed = True
        for r in (self.reply, self.spec):
            if r:
                await self._cancel_reply(r)
        if self._stt_task:
            self._stt_task.cancel()
        with contextlib.suppress(Exception):
            await self.stt.close()
        with contextlib.suppress(Exception):
            await self.tts.close()
        summary = self.metrics.close(reason)
        with contextlib.suppress(Exception):
            await self.send({"type": "call_summary", "summary": summary})
        return summary

    async def send(self, msg: dict[str, Any]) -> None:
        if not self._closed or msg.get("type") == "call_summary":
            await self._send_json(msg)

    # -- inputs from the browser -----------------------------------------
    async def on_mic_audio(self, pcm: bytes) -> None:
        self.metrics.counts["mic_audio_bytes"] += len(pcm)
        await self.stt.send_audio(pcm)

    async def on_client_message(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        turn_idx = msg.get("turn")
        turn = self.metrics.turns[turn_idx] if isinstance(turn_idx, int) and 0 <= turn_idx < len(self.metrics.turns) else None
        if kind == "client_metric" and turn:
            data = {k: v for k, v in msg.items() if k not in {"type", "turn"} and isinstance(v, (int, float))}
            turn.client.update(data)
            self.metrics.event("client_metric", turn=turn.index, **data)
        elif kind == "playback_done":
            self.metrics.event("client_playback_done", turn=turn_idx)
            if self.client_playing_turn == turn_idx:
                self.client_playing_turn = None
        elif kind == "client_info":
            self.metrics.event("client_info", **{k: v for k, v in msg.items() if k != "type"})

    # -- STT events ------------------------------------------------------
    async def _read_stt(self) -> None:
        try:
            async for ev in self.stt.events():
                await self.on_stt_event(ev)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.metrics.error("stt", repr(e), self.user_turn)
            await self.send({"type": "error", "where": "stt", "message": str(e)})

    async def on_stt_event(self, ev: dict[str, Any]) -> None:
        kind = ev.get("type", "")
        text = ev.get("transcript", "")
        if kind == "turn.start":
            if self._bot_is_talking():
                await self._barge_in()
            self.user_turn = self.metrics.new_turn()
            self.metrics.mark(self.user_turn, "stt_turn_start")
            await self.send({"type": "user_turn_start", "turn": self.user_turn.index})
            return
        if kind == "error":
            self.metrics.error("stt", f"{ev.get('title')}: {ev.get('message')}", self.user_turn)
            await self.send({"type": "error", "where": "stt", "message": ev.get("message")})
            return
        if kind == "connected":
            self.metrics.event("stt_ready", request_id=ev.get("request_id"))
            return

        turn = self.user_turn
        if turn is None:  # STT sent turn events without turn.start; open a turn anyway
            turn = self.user_turn = self.metrics.new_turn()
            self.metrics.mark(turn, "stt_turn_start", synthetic=True)

        if kind == "turn.update":
            turn.counts["stt_partials"] += 1
            self.metrics.mark(turn, "stt_first_partial", transcript=text)
            await self.send({"type": "partial", "turn": turn.index, "text": text})
        elif kind == "turn.eager_end":
            self.metrics.mark(turn, "stt_eager_end", transcript=text)
            if self.s.speculative_llm and text.strip():
                await self._start_speculation(turn, text)
        elif kind == "turn.resume":
            self.metrics.event("stt_resume", turn=turn.index)
            if self.spec:
                await self._discard_spec(turn, "resume")
        elif kind == "turn.end":
            self.metrics.mark(turn, "stt_turn_end", transcript=text)
            await self._on_user_turn_end(turn, text)

    # -- turn handling ---------------------------------------------------
    def _bot_is_talking(self) -> bool:
        if self.reply and self.reply.active():
            return True
        return self.client_playing_turn is not None and self.metrics.now_ms() < self._playback_end_ms

    async def _barge_in(self) -> None:
        r = self.reply
        self.metrics.counts["barge_ins"] += 1
        cleared = r.turn.index if r else self.client_playing_turn
        self.metrics.event("barge_in", turn=cleared, spoken=r.spoken if r else None)
        await self.send({"type": "clear", "turn": cleared})
        self.client_playing_turn = None
        if r:
            r.turn.interrupted = True
            await self._cancel_reply(r)
            if not r.committed and r.llm:
                self.history += r.llm.interrupted_messages(r.spoken)
                r.committed = True
            self.reply = None

    def _make_reply(self, turn: TurnMetrics, text: str) -> Reply:
        reply = Reply(turn=turn, transcript=text)
        reply.llm = LLMTurn(
            self.llm_client,
            model=self.s.llm_model,
            effort=self.s.llm_effort,
            max_tokens=self.s.llm_max_tokens,
            system=self.system,
            clinic=self.clinic,
            history=self.history,
            user_text=text,
            on_event=lambda event, /, **d: self.metrics.mark(turn, event, **d),
            confirmed=reply.confirmed,
        )
        reply.producer = asyncio.create_task(self._produce(reply))
        return reply

    async def _start_speculation(self, turn: TurnMetrics, text: str) -> None:
        if self.spec:
            if _norm(self.spec.transcript) == _norm(text):
                return
            await self._discard_spec(turn, "new_eager_end")
        turn.counts["speculative_starts"] = turn.counts.get("speculative_starts", 0) + 1
        self.metrics.event("speculation_start", turn=turn.index, transcript=text)
        self.spec = self._make_reply(turn, text)

    async def _on_user_turn_end(self, turn: TurnMetrics, text: str) -> None:
        if not text.strip():
            self.metrics.event("empty_turn", turn=turn.index)
            if self.spec:
                await self._discard_spec(turn, "empty_turn")
            return
        turn.user_text = text
        await self.send({"type": "turn_end", "turn": turn.index, "text": text})
        if self.reply and self.reply.active():  # should not happen: turn.start barges in
            await self._barge_in()

        if self.spec and _norm(self.spec.transcript) == _norm(text):
            turn.speculative = "hit"
            self.metrics.event("speculation_hit", turn=turn.index)
            reply, self.spec = self.spec, None
        else:
            if self.spec:
                await self._discard_spec(turn, "transcript_changed")
            if turn.counts.get("speculative_starts"):
                turn.speculative = "miss"
            elif self.s.speculative_llm:
                turn.speculative = "not_triggered"
            reply = self._make_reply(turn, text)
        reply.confirmed.set()
        self._start_speaking(reply)

    async def _discard_spec(self, turn: TurnMetrics, reason: str) -> None:
        assert self.spec
        self.metrics.event("speculation_discarded", turn=turn.index, reason=reason)
        await self._cancel_reply(self.spec)
        self.spec = None
        # Marks from discarded work would skew the timings; the next run sets fresh ones.
        for k in [k for k in turn.marks if k.startswith("llm_")]:
            del turn.marks[k]

    def _start_speaking(self, reply: Reply) -> None:
        self.reply = reply
        reply.speaker = asyncio.create_task(self._speak(reply))

    async def _cancel_reply(self, reply: Reply) -> None:
        for t in (reply.speaker, reply.producer):
            if t and not t.done():
                t.cancel()
        for t in (reply.speaker, reply.producer):
            if t:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t

    # -- LLM -> phrases --------------------------------------------------
    async def _produce(self, reply: Reply) -> None:
        turn, llm = reply.turn, reply.llm
        assert llm
        self.metrics.mark(turn, "llm_start", speculative=not reply.confirmed.is_set())
        try:
            async for phrase in llm.phrases():
                self.metrics.mark(turn, "llm_first_phrase", text=phrase)
                reply.queue.put_nowait(phrase)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.metrics.error("llm", repr(e), turn)
            await self.send({"type": "error", "where": "llm", "message": str(e)})
            if not llm.text_out:
                reply.queue.put_nowait(APOLOGY)
        reply.queue.put_nowait(None)
        self.metrics.mark(turn, "llm_end", rounds=llm.rounds)

        await reply.confirmed.wait()
        for k, v in llm.usage.items():
            turn.counts[k] += v
        turn.counts["llm_rounds"] += llm.rounds
        turn.tool_calls += llm.tool_calls
        turn.bot_text = llm.text_out
        if not reply.committed:
            self.history += llm.new_messages
            reply.committed = True
        if llm.booking:
            self.metrics.outcome["booking"] = llm.booking
            self.metrics.event("booking", turn=turn.index, **llm.booking)
            await self.send({"type": "booking", "booking": llm.booking})
        await self.send({"type": "bot_text", "turn": turn.index, "text": llm.text_out})

    # -- phrases -> TTS -> browser ---------------------------------------
    async def _speak(self, reply: Reply) -> None:
        turn = reply.turn
        ctx = self.tts.new_context()
        forward = asyncio.create_task(self._forward_audio(reply, ctx))
        try:
            async for phrase in reply.phrases():
                self.metrics.mark(turn, "tts_first_send")
                await ctx.send(phrase)
                turn.counts["tts_chars"] += len(phrase)
                turn.counts["tts_chunks_sent"] += 1
                reply.spoken += phrase
            await ctx.end()
            await forward
        except asyncio.CancelledError:
            forward.cancel()
            with contextlib.suppress(Exception):
                await ctx.cancel()
            raise
        except Exception as e:
            forward.cancel()
            self.metrics.error("tts", repr(e), turn)
            await self.send({"type": "error", "where": "tts", "message": str(e)})

    async def _forward_audio(self, reply: Reply, ctx: Any) -> None:
        turn = reply.turn
        header = struct.pack("<I", turn.index)
        async for pcm in ctx.audio():
            if "tts_first_audio" not in turn.marks:
                self.metrics.mark(turn, "tts_first_audio")
                self.client_playing_turn = turn.index
                self._playback_end_ms = self.metrics.now_ms() + 250  # browser scheduling margin
                await self.send({"type": "bot_audio_start", "turn": turn.index,
                                 "sample_rate": self.s.tts_sample_rate})
            turn.counts["tts_audio_bytes"] += len(pcm)
            chunk_ms = len(pcm) / 2 / self.s.tts_sample_rate * 1000
            self._playback_end_ms = max(self._playback_end_ms, self.metrics.now_ms()) + chunk_ms
            await self._send_bytes(header + pcm)
        self.metrics.mark(turn, "tts_done")
        await self.send({"type": "bot_audio_end", "turn": turn.index})


def parse_client_text(raw: str) -> dict[str, Any]:
    try:
        msg = json.loads(raw)
        return msg if isinstance(msg, dict) else {}
    except json.JSONDecodeError:
        return {}
