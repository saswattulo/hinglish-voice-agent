import asyncio
import dataclasses
import datetime as dt
import json

import pytest

from app.clinic import Clinic
from app.config import Settings
from app.session import CallSession

from .fakes import FakeClaude, FakeSTT, FakeTTS, text_reply, tool_reply

MONDAY = dt.date(2026, 11, 2)


class Harness:
    def __init__(self, tmp_path, replies, speculative=True, token_delay=0.0, tts_delay=0.0):
        self.settings = dataclasses.replace(
            Settings(), metrics_dir=tmp_path, speculative_llm=speculative,
            cartesia_api_key="x", anthropic_api_key="y",
        )
        self.json: list[dict] = []
        self.audio: list[bytes] = []
        self.stt = FakeSTT()
        self.tts = FakeTTS(delay=tts_delay)
        self.claude = FakeClaude(replies, token_delay=token_delay)

        async def send_json(m):
            self.json.append(m)

        async def send_bytes(b):
            self.audio.append(b)

        self.session = CallSession(
            settings=self.settings, send_json=send_json, send_bytes=send_bytes,
            stt=self.stt, tts=self.tts, llm_client=self.claude, clinic=Clinic(today=MONDAY),
        )

    async def stt_event(self, type_, transcript=None):
        ev = {"type": type_}
        if transcript is not None:
            ev["transcript"] = transcript
        await self.stt.q.put(ev)
        await asyncio.sleep(0.01)

    async def user_says(self, text, eager=None):
        await self.stt_event("turn.start")
        await self.stt_event("turn.update", text)
        if eager is not None:
            await self.stt_event("turn.eager_end", eager)
        await self.stt_event("turn.end", text)

    async def settle(self):
        for _ in range(50):
            r = self.session.reply
            if not (r and r.active()):
                break
            await asyncio.sleep(0.01)

    def types(self):
        return [m["type"] for m in self.json]


async def test_greeting_then_one_turn_is_fully_measured(tmp_path):
    h = Harness(tmp_path, [text_reply("जी, बताइए किस doctor से मिलना है?")])
    await h.session.start()
    await h.settle()
    await asyncio.sleep(0.3)  # let the greeting finish "playing"
    await h.user_says("mujhe appointment chahiye")
    await h.settle()
    summary = await h.session.close("hangup")

    assert [t["index"] for t in summary["turns"]] == [0, 1]
    turn = summary["turns"][1]
    for mark in ("stt_turn_start", "stt_first_partial", "stt_turn_end", "llm_start", "llm_first_token",
                 "llm_first_phrase", "llm_end", "tts_first_send", "tts_first_audio", "tts_done"):
        assert mark in turn["marks_ms"], mark
    d = turn["derived_ms"]
    assert d["turn_end_to_first_audio_ms"] is not None and d["turn_end_to_first_audio_ms"] >= 0
    assert turn["counts"]["input_tokens"] == 100 and turn["counts"]["output_tokens"] == 20
    assert turn["counts"]["tts_chars"] > 0 and turn["counts"]["tts_audio_bytes"] > 0
    assert turn["speculative"] == "not_triggered"
    assert summary["totals"]["barge_ins"] == 0 and not turn["interrupted"]
    assert summary["derived_ms"]["greeting_first_audio_ms"] is not None

    # History holds the user turn and Claude's answer, in order.
    roles = [m["role"] for m in h.session.history]
    assert roles == ["user", "assistant"]

    # Audio frames carry the turn index.
    assert {int.from_bytes(b[:4], "little") for b in h.audio} == {0, 1}

    # Files on disk.
    calls = tmp_path / "calls"
    events = (calls / f"{summary['call_id']}.events.jsonl").read_text().splitlines()
    names = [json.loads(e)["event"] for e in events]
    assert names[0] == "call_start" and names[-1] == "call_end"
    assert (calls / f"{summary['call_id']}.summary.json").exists()
    index = (tmp_path / "calls_index.jsonl").read_text().splitlines()
    assert json.loads(index[-1])["call_id"] == summary["call_id"]


async def test_speculative_hit_reuses_early_llm_work(tmp_path):
    h = Harness(tmp_path, [text_reply("ठीक है, कल सुबह दस बजे free है।")], token_delay=0.005)
    await h.session.start()
    await h.settle()
    await h.user_says("kal subah", eager="kal subah")
    await h.settle()
    summary = await h.session.close("hangup")

    turn = summary["turns"][1]
    assert turn["speculative"] == "hit"
    assert len(h.claude.requests) == 1
    # LLM started before STT confirmed the turn was over.
    assert turn["marks_ms"]["llm_start"] <= turn["marks_ms"]["stt_turn_end"]
    assert summary["totals"]["speculative_hits"] == 1


async def test_speculative_miss_discards_and_restarts(tmp_path):
    h = Harness(tmp_path, [text_reply("wrong answer"), text_reply("सही जवाब।")], token_delay=0.01)
    await h.session.start()
    await h.settle()
    await h.stt_event("turn.start")
    await h.stt_event("turn.eager_end", "kal")
    await h.stt_event("turn.resume")
    await h.stt_event("turn.end", "kal shaam ko")
    await h.settle()
    summary = await h.session.close("hangup")

    turn = summary["turns"][1]
    assert turn["speculative"] == "miss"
    assert turn["bot_text"].strip() == "सही जवाब।"
    assert h.session.history[0]["content"] == "kal shaam ko"
    assert turn["marks_ms"]["llm_start"] >= turn["marks_ms"]["stt_turn_end"]


async def test_barge_in_stops_bot_and_records_what_was_heard(tmp_path):
    long = "यह एक बहुत लंबा जवाब है। " * 20
    h = Harness(tmp_path, [text_reply(long), text_reply("जी।")], speculative=False, token_delay=0.005)
    await h.session.start()
    await h.settle()
    await h.user_says("doctor kaun hai")
    await asyncio.sleep(0.05)  # bot is mid-reply
    await h.user_says("ruko ruko")
    await h.settle()
    summary = await h.session.close("hangup")

    assert "clear" in h.types()
    assert summary["totals"]["barge_ins"] >= 1
    first = summary["turns"][1]
    assert first["interrupted"] is True
    # History keeps alternating roles and marks the cut-off reply.
    roles = [m["role"] for m in h.session.history]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert "[interrupted by caller]" in h.session.history[1]["content"]


async def test_booking_tool_runs_and_is_recorded(tmp_path):
    h = Harness(tmp_path, [
        tool_reply("book_appointment", {"doctor": "Dr. Sharma", "date": "2026-11-03", "time": "10:00",
                                        "patient_name": "Ravi", "phone": "98450 12345"}),
        text_reply("आपका appointment book हो गया, booking ID SJV-101 है।"),
    ])
    await h.session.start()
    await h.settle()
    await h.user_says("haan confirm kar dijiye")
    await h.settle()
    await asyncio.sleep(0.02)
    summary = await h.session.close("hangup")

    assert summary["outcome"]["booking"]["booking_id"] == "SJV-101"
    turn = summary["turns"][1]
    assert turn["counts"]["llm_rounds"] == 2
    assert turn["tool_calls"][0]["name"] == "book_appointment" and turn["tool_calls"][0]["ok"]
    assert "booking" in h.types()
    # user, assistant(tool_use), user(tool_result), assistant(text)
    assert [m["role"] for m in h.session.history] == ["user", "assistant", "user", "assistant"]


async def test_speculative_run_cannot_book_before_turn_is_confirmed(tmp_path):
    h = Harness(tmp_path, [
        tool_reply("book_appointment", {"doctor": "Dr. Sharma", "date": "2026-11-03", "time": "10:00",
                                        "patient_name": "Ravi", "phone": "9845012345"}),
    ])
    await h.session.start()
    await h.settle()
    await h.stt_event("turn.start")
    await h.stt_event("turn.eager_end", "haan")
    await asyncio.sleep(0.05)
    assert not h.session.clinic.booked  # waiting for confirmation
    await h.stt_event("turn.resume")
    await asyncio.sleep(0.02)
    assert not h.session.clinic.booked
    await h.session.close("hangup")


async def test_client_metrics_are_attached_to_turns(tmp_path):
    h = Harness(tmp_path, [text_reply("जी।")])
    await h.session.start()
    await h.settle()
    await h.user_says("hello")
    await h.settle()
    await h.session.on_client_message({"type": "client_metric", "turn": 1, "turn_end_to_playback_ms": 812})
    await h.session.on_mic_audio(b"\x00\x00" * 1600)
    summary = await h.session.close("hangup")
    assert summary["turns"][1]["derived_ms"]["client_turn_end_to_playback_ms"] == 812
    assert summary["totals"]["mic_audio_ms"] == 100.0


async def test_llm_failure_is_logged_and_bot_apologises(tmp_path):
    h = Harness(tmp_path, [])

    def boom(**kwargs):
        raise RuntimeError("overloaded")

    h.claude.beta.messages.stream = boom
    await h.session.start()
    await h.settle()
    await h.user_says("hello")
    await h.settle()
    summary = await h.session.close("hangup")
    turn = summary["turns"][1]
    assert turn["errors"] and turn["errors"][0]["where"] == "llm"
    assert "माफ़" in h.tts.contexts[-1].texts[0]
