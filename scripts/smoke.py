"""Live check of every external service with real keys, with timings.

    python -m scripts.smoke

1. Sonic TTS speaks a Hinglish sentence (saved to logs/smoke/tts.wav).
2. Ink STT transcribes that same audio, streamed in real time, and ends the turn.
3. Claude answers one caller line with the clinic prompt and tools.

Results go to logs/smoke/<timestamp>.json. Exit code is non-zero if any step fails.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from app.cartesia import CartesiaSTT, CartesiaTTS
from app.clinic import Clinic, system_prompt
from app.config import settings
from app.llm import LLMTurn

SENTENCE = "नमस्ते, मुझे कल सुबह Dr. Sharma के साथ appointment चाहिए।"
CALLER_LINE = "Mujhe kal subah Dr. Sharma ke saath appointment chahiye."
STT_RATE = 16000


def ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000, 1)


async def check_tts(out_dir: Path) -> tuple[dict, bytes]:
    tts = CartesiaTTS(
        settings.cartesia_api_key, model=settings.tts_model, voice_id=settings.tts_voice_id,
        language=settings.tts_language, sample_rate=STT_RATE, speed=settings.tts_speed,
        emotion=settings.tts_emotion, max_buffer_delay_ms=settings.tts_max_buffer_delay_ms,
    )
    t0 = time.monotonic()
    await tts.connect()
    res: dict = {"connect_ms": ms(t0)}
    ctx = tts.new_context()
    t1 = time.monotonic()
    await ctx.send(SENTENCE)
    await ctx.end()
    pcm = bytearray()
    async for chunk in ctx.audio():
        if not pcm:
            res["first_audio_ms"] = ms(t1)
        pcm += chunk
    res["total_ms"] = ms(t1)
    res["audio_ms"] = round(len(pcm) / 2 / STT_RATE * 1000, 1)
    await tts.close()

    with wave.open(str(out_dir / "tts.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(STT_RATE)
        w.writeframes(bytes(pcm))
    return res, bytes(pcm)


async def check_stt(pcm: bytes) -> dict:
    stt = CartesiaSTT(
        settings.cartesia_api_key, model=settings.stt_model, sample_rate=STT_RATE,
        turn_end_timeout_ms=settings.stt_turn_end_timeout_ms, keyterms=settings.stt_keyterms,
    )
    t0 = time.monotonic()
    await stt.connect()
    res: dict = {"connect_ms": ms(t0), "events": []}
    chunk = STT_RATE * 2 // 10  # 100 ms
    audio_done_at: list[float] = []

    async def feed() -> None:
        for i in range(0, len(pcm), chunk):
            await stt.send_audio(pcm[i:i + chunk])
            await asyncio.sleep(0.1)
        audio_done_at.append(time.monotonic())
        silence = b"\x00" * chunk
        for _ in range(int(settings.stt_turn_end_timeout_ms / 100) + 20):
            await stt.send_audio(silence)
            await asyncio.sleep(0.1)

    feeder = asyncio.create_task(feed())
    try:
        async with asyncio.timeout(len(pcm) / 2 / STT_RATE + settings.stt_turn_end_timeout_ms / 1000 + 5):
            async for ev in stt.events():
                kind = ev.get("type")
                res["events"].append(kind)
                if kind == "error":
                    raise RuntimeError(f"STT error: {ev}")
                if kind == "turn.eager_end" and audio_done_at:
                    res.setdefault("audio_end_to_eager_end_ms", ms(audio_done_at[0]))
                if kind == "turn.end":
                    res["transcript"] = ev.get("transcript", "")
                    if audio_done_at:
                        res["audio_end_to_turn_end_ms"] = ms(audio_done_at[0])
                    break
    finally:
        feeder.cancel()
        await stt.close()
    if "transcript" not in res:
        raise RuntimeError(f"STT never ended the turn; events: {res['events']}")
    return res


async def check_llm() -> dict:
    clinic = Clinic()
    marks: dict[str, float] = {}
    t0 = time.monotonic()

    def on_event(name: str, /, **_: object) -> None:
        marks.setdefault(name, ms(t0))

    turn = LLMTurn(
        anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key),
        model=settings.llm_model, effort=settings.llm_effort, max_tokens=settings.llm_max_tokens,
        system=system_prompt(clinic.today), clinic=clinic, history=[], user_text=CALLER_LINE,
        on_event=on_event, confirmed=asyncio.Event(),
    )
    phrases = [p async for p in turn.phrases()]
    return {
        "model": settings.llm_model,
        "effort": settings.llm_effort,
        "first_token_ms": marks.get("llm_first_token"),
        "total_ms": ms(t0),
        "rounds": turn.rounds,
        "tools": [c["name"] for c in turn.tool_calls],
        "usage": turn.usage,
        "first_phrase": phrases[0] if phrases else "",
        "reply": turn.text_out,
    }


async def main() -> int:
    missing = settings.missing_keys()
    if missing:
        print(f"Missing {', '.join(missing)} in .env")
        return 2
    out_dir = settings.metrics_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"at": datetime.now(timezone.utc).isoformat(), "ok": True}
    pcm = b""

    for name in ("tts", "stt", "llm"):
        try:
            if name == "tts":
                report["tts"], pcm = await check_tts(out_dir)
            elif name == "stt":
                if not pcm:
                    raise RuntimeError("skipped: no TTS audio to transcribe")
                report["stt"] = await check_stt(pcm)
            else:
                report["llm"] = await check_llm()
            print(f"✓ {name}: {json.dumps(report[name], ensure_ascii=False)}")
        except Exception as e:
            report["ok"] = False
            report[name] = {"error": repr(e)}
            print(f"✗ {name}: {e!r}")

    path = out_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
