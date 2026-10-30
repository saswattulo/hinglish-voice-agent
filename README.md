# Hinglish Voice Agent

A browser voice agent that books appointments at a (pretend) Bangalore clinic, in Hinglish.
It runs on Cartesia for both ears and mouth, with Claude in between, and it measures every
step of every turn so the delay can be shown and tuned.

```
 browser mic ──16 kHz PCM──▶ FastAPI /ws ──▶ Cartesia Ink-2 STT (turn detection)
                                   │                │ turn.eager_end / turn.end
                                   │                ▼
                                   │           Claude (streamed, clinic tools)
                                   │                │ phrases
                                   │                ▼
 browser speaker ◀──24 kHz PCM──── │ ◀──── Cartesia Sonic 3.6 TTS (one context per reply)
```

## What it does

- **Hinglish by design.** The receptionist ("Priya") answers in Hindi word order with
  everyday English words, writing Hindi in Devanagari and English in Latin script so Sonic
  pronounces both naturally. STT keyterms help Ink catch doctor names.
- **Real booking flow.** Claude calls `check_availability` and `book_appointment` against an
  in-memory clinic (three doctors, Monday to Saturday slots). Booking only runs after the
  caller confirms.
- **Low-latency turn taking.**
  - Ink's turn detection decides when the caller is done.
  - *Speculative start:* Claude starts on Ink's `turn.eager_end`. If the final transcript
    matches, that head start is kept; if the caller keeps talking, the work is thrown away.
    Side-effect tools (booking) wait until the turn is confirmed, so speculation can never
    book anything.
  - Claude's output is cut into phrases (first one early, at a comma) and streamed into a
    single Sonic context with `continue: true`, so speech starts before the reply is written.
  - *Barge-in:* if the caller talks over the bot, TTS is cancelled, the browser drops queued
    audio, and history records only what the caller actually heard.
- **Metrics from start to end of every call** (see below), plus a dashboard at `/dashboard`.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env        # add CARTESIA_API_KEY and ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000, press **Start call**, and allow the microphone. Headphones give
the cleanest barge-in. `localhost` counts as a secure origin, so the mic works without HTTPS.

Pick a voice that suits Hinglish in the Cartesia playground and set `TTS_VOICE_ID`.
`TTS_SPEED` (0.6 to 1.5) and `TTS_EMOTION` tune delivery.

Tests run fully offline with fake Cartesia and Claude clients:

```bash
pytest
```

## Metrics

Each call writes, under `logs/calls/`:

| File | What's in it |
|---|---|
| `<call_id>.events.jsonl` | Every event with a millisecond timestamp, flushed as it happens: connects, each STT event, speculation start/hit/discard, each LLM round with stop reason and token usage, each tool call with arguments, result and duration, TTS first send/first audio/done, barge-ins, browser-reported playback, errors. |
| `<call_id>.summary.json` | The call: config used, connect times, totals, booking outcome, and every turn with its marks, derived latencies and counts. |

and `logs/calls_index.jsonl` gets one line per finished call.

Per turn, derived from the marks:

| Metric | Meaning |
|---|---|
| `turn_end_to_first_audio_ms` | Server voice-to-voice: STT says the caller is done → first bot audio leaves the server. |
| `client_turn_end_to_playback_ms` | Browser voice-to-voice: browser hears "turn ended" → bot audio starts playing. |
| `llm_ttft_ms`, `llm_first_phrase_ms`, `llm_total_ms` | Claude: first token, first speakable phrase, whole reply (across tool rounds). |
| `tts_ttfa_ms`, `tts_total_ms` | Sonic: first text sent → first audio back; → done. |
| `stt_eager_to_end_ms` | How much head start the eager end signal gave. |
| `user_speech_ms`, `bot_audio_ms` | How long each side spoke. |
| counts | input/output/cache tokens, LLM rounds, TTS characters and chunks, audio bytes, STT partials, playback underruns. |
| flags | speculative `hit` / `miss` / `not_triggered`, `interrupted`, errors. |

Look at them in the browser at http://localhost:8000/dashboard (p50/p95 cards, per-call
table, per-turn waterfall), through `GET /api/calls` and `GET /api/calls/{id}`, or in the
terminal:

```bash
python -m scripts.report --last 20
```

## Layout

```
app/
  main.py      FastAPI app: call page, dashboard, /ws, /api/calls
  session.py   one call: turn taking, speculation, barge-in, metrics hooks
  cartesia.py  Ink STT (turns endpoint) and Sonic TTS WebSocket clients
  llm.py       Claude streaming, tool loop, phrase chunker
  clinic.py    clinic data, tools, system prompt
  metrics.py   per-call / per-turn recorder, persistence, percentiles
static/        call page, mic AudioWorklet, dashboard
scripts/       report.py
tests/         offline tests with fake STT, TTS and Claude
```

## Configuration

All settings live in `.env` (see `.env.example`). The ones that matter most for latency:

- `LLM_MODEL` / `LLM_EFFORT`: defaults to `claude-opus-5-5` at `low` effort. A smaller
  model such as `claude-haiku-5-5` is worth comparing; the dashboard shows the difference.
- `SPECULATIVE_LLM`: turn the eager-end head start on or off to measure what it saves.
- `STT_TURN_END_TIMEOUT_MS`: upper bound on how long Ink waits after speech stops.
- `TTS_MAX_BUFFER_DELAY_MS`: how long Sonic may wait for more text before speaking.

Claude requests opt into server-side refusal fallbacks (`fallbacks: "default"`).

## Next steps

- Phone calls: plug a telephony provider's media stream into the same `CallSession`
  (8 kHz μ-law in and out are both supported by Cartesia).
- Docker image and a one-command cloud deploy.
- An eval set of recorded Hinglish calls, scored automatically, run in CI with latency budgets.
