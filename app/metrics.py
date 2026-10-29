"""Per-call and per-turn metrics, persisted to disk as they happen.

Every call gets two files under ``<metrics_dir>/calls/``:

* ``<call_id>.events.jsonl``: one line per event, written immediately, so a crash
  mid-call still leaves a usable timeline.
* ``<call_id>.summary.json``: the call summary with every turn's timings and counts,
  written when the call ends.

A one-line summary of each finished call is also appended to
``<metrics_dir>/calls_index.jsonl`` so reports can scan calls quickly.

All ``*_ms`` marks are milliseconds since the call started, from a monotonic clock.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _diff(marks: dict[str, float], end: str, start: str) -> float | None:
    if end in marks and start in marks:
        return round(marks[end] - marks[start], 1)
    return None


@dataclass
class TurnMetrics:
    index: int
    marks: dict[str, float] = field(default_factory=dict)
    user_text: str = ""
    bot_text: str = ""
    counts: dict[str, int] = field(
        default_factory=lambda: {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "llm_rounds": 0,
            "tts_chars": 0,
            "tts_chunks_sent": 0,
            "tts_audio_bytes": 0,
            "stt_partials": 0,
        }
    )
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    speculative: str = "off"  # off | hit | miss | not_triggered
    interrupted: bool = False
    client: dict[str, float] = field(default_factory=dict)

    def derived(self, tts_sample_rate: int) -> dict[str, float | None]:
        m = self.marks
        audio_ms = None
        if self.counts["tts_audio_bytes"]:
            audio_ms = round(self.counts["tts_audio_bytes"] / 2 / tts_sample_rate * 1000, 1)
        return {
            # How long the user spoke, as STT saw it.
            "user_speech_ms": _diff(m, "stt_turn_end", "stt_turn_start"),
            # Gap between STT's "maybe done" and "done" signals.
            "stt_eager_to_end_ms": _diff(m, "stt_turn_end", "stt_eager_end"),
            "llm_ttft_ms": _diff(m, "llm_first_token", "llm_start"),
            "llm_first_phrase_ms": _diff(m, "llm_first_phrase", "llm_start"),
            "llm_total_ms": _diff(m, "llm_end", "llm_start"),
            "tts_ttfa_ms": _diff(m, "tts_first_audio", "tts_first_send"),
            "tts_total_ms": _diff(m, "tts_done", "tts_first_send"),
            # Server-side voice-to-voice: user finished -> first bot audio leaves the server.
            "turn_end_to_first_audio_ms": _diff(m, "tts_first_audio", "stt_turn_end"),
            "turn_end_to_llm_start_ms": _diff(m, "llm_start", "stt_turn_end"),
            "bot_audio_ms": audio_ms,
            # Client-side voice-to-voice: browser got turn_end -> browser started playing.
            "client_turn_end_to_playback_ms": self.client.get("turn_end_to_playback_ms"),
        }

    def to_dict(self, tts_sample_rate: int) -> dict[str, Any]:
        return {
            "index": self.index,
            "user_text": self.user_text,
            "bot_text": self.bot_text,
            "marks_ms": self.marks,
            "derived_ms": self.derived(tts_sample_rate),
            "counts": self.counts,
            "tool_calls": self.tool_calls,
            "speculative": self.speculative,
            "interrupted": self.interrupted,
            "client": self.client,
            "errors": self.errors,
        }


class CallMetrics:
    def __init__(self, metrics_dir: Path, config: dict[str, Any], tts_sample_rate: int):
        self.call_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.started_at = _now_iso()
        self._t0 = time.monotonic()
        self._tts_sample_rate = tts_sample_rate
        self.config = config
        self.marks: dict[str, float] = {}
        self.counts: dict[str, int] = {"mic_audio_bytes": 0, "barge_ins": 0}
        self.errors: list[dict[str, Any]] = []
        self.turns: list[TurnMetrics] = []
        self.outcome: dict[str, Any] = {}
        self.end_reason = ""
        self._closed = False

        self._dir = metrics_dir / "calls"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = metrics_dir / "calls_index.jsonl"
        self._events = open(self._dir / f"{self.call_id}.events.jsonl", "a", encoding="utf-8")
        self.event("call_start", config=config)

    # -- clock -----------------------------------------------------------
    def now_ms(self) -> float:
        return round((time.monotonic() - self._t0) * 1000, 1)

    # -- recording -------------------------------------------------------
    def event(self, name: str, /, turn: int | None = None, **data: Any) -> float:
        t = self.now_ms()
        record = {"t_ms": t, "wall": _now_iso(), "event": name}
        if turn is not None:
            record["turn"] = turn
        record.update(data)
        if not self._closed:
            self._events.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._events.flush()
        return t

    def mark_call(self, name: str, /, **data: Any) -> None:
        self.marks.setdefault(name, self.event(name, **data))

    def new_turn(self) -> TurnMetrics:
        turn = TurnMetrics(index=len(self.turns))
        self.turns.append(turn)
        return turn

    def mark(self, turn: TurnMetrics, name: str, /, **data: Any) -> None:
        """Record ``name`` on ``turn`` the first time it happens."""
        t = self.event(name, turn=turn.index, **data)
        turn.marks.setdefault(name, t)

    def error(self, where: str, message: str, turn: TurnMetrics | None = None) -> None:
        err = {"t_ms": self.now_ms(), "where": where, "message": message}
        self.event("error", turn=turn.index if turn else None, where=where, message=message)
        (turn.errors if turn else self.errors).append(err)

    # -- summary ---------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        turns = [t.to_dict(self._tts_sample_rate) for t in self.turns if t.user_text or t.bot_text]
        totals = {
            k: sum(t["counts"][k] for t in turns)
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                      "cache_creation_input_tokens", "tts_chars", "llm_rounds")
        }
        totals["tool_calls"] = sum(len(t["tool_calls"]) for t in turns)
        totals["turn_errors"] = sum(len(t["errors"]) for t in turns)
        totals["interrupted_turns"] = sum(1 for t in turns if t["interrupted"])
        totals["speculative_hits"] = sum(1 for t in turns if t["speculative"] == "hit")
        totals["speculative_misses"] = sum(1 for t in turns if t["speculative"] == "miss")
        totals["mic_audio_ms"] = round(self.counts["mic_audio_bytes"] / 2 / 16000 * 1000, 1)
        totals["bot_audio_ms"] = round(sum(t["derived_ms"]["bot_audio_ms"] or 0 for t in turns), 1)
        totals["barge_ins"] = self.counts["barge_ins"]
        return {
            "call_id": self.call_id,
            "started_at": self.started_at,
            "duration_ms": self.now_ms(),
            "end_reason": self.end_reason,
            "config": self.config,
            "marks_ms": self.marks,
            "derived_ms": {
                "stt_connect_ms": _diff(self.marks, "stt_connected", "stt_connect_start"),
                "tts_connect_ms": _diff(self.marks, "tts_connected", "tts_connect_start"),
                "greeting_first_audio_ms": self.turns[0].marks.get("tts_first_audio") if self.turns else None,
            },
            "totals": totals,
            "outcome": self.outcome,
            "errors": self.errors,
            "turns": turns,
        }

    def close(self, reason: str) -> dict[str, Any]:
        if self._closed:
            return self.summary()
        self.end_reason = reason
        self.event("call_end", reason=reason)
        data = self.summary()
        self._closed = True
        self._events.close()
        (self._dir / f"{self.call_id}.summary.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        index_line = {
            "call_id": self.call_id,
            "started_at": self.started_at,
            "duration_ms": data["duration_ms"],
            "end_reason": reason,
            "turns": len(data["turns"]),
            "booked": bool(self.outcome.get("booking")),
            "totals": data["totals"],
            "turn_end_to_first_audio_ms": [
                t["derived_ms"]["turn_end_to_first_audio_ms"] for t in data["turns"]
            ],
        }
        with open(self._index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(index_line, ensure_ascii=False, default=str) + "\n")
        return data


def percentile(values: list[float], p: float) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 1)


LATENCY_KEYS = (
    "turn_end_to_first_audio_ms",
    "client_turn_end_to_playback_ms",
    "llm_ttft_ms",
    "llm_first_phrase_ms",
    "llm_total_ms",
    "tts_ttfa_ms",
    "stt_eager_to_end_ms",
)


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """p50/p95 for each latency across all turns of the given calls."""
    turns = [t for s in summaries for t in s.get("turns", [])]
    out: dict[str, Any] = {"calls": len(summaries), "turns": len(turns)}
    for key in LATENCY_KEYS:
        vals = [t["derived_ms"].get(key) for t in turns]
        vals = [v for v in vals if v is not None]
        out[key] = {"n": len(vals), "p50": percentile(vals, 50), "p95": percentile(vals, 95)}
    return out


def load_summaries(metrics_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    paths = sorted((metrics_dir / "calls").glob("*.summary.json"), reverse=True)
    if limit:
        paths = paths[:limit]
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]
