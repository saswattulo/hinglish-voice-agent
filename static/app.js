// Browser side of a call: mic -> server as 16 kHz PCM16, server audio -> speakers.
// Also measures what only the browser can see (when audio actually starts playing)
// and reports it back so it lands in the call's metrics.

const $ = (id) => document.getElementById(id);
const callBtn = $("callBtn"), statusEl = $("status"), logEl = $("log"), partialEl = $("partial");
const meterBar = document.querySelector("#meter span");

let ws = null, ctx = null, micStream = null, micNode = null, srcNode = null;
let nextPlayTime = 0;
const turns = new Map(); // turn index -> playback state
const cleared = new Set();

function setStatus(s) { statusEl.textContent = s; }

function addMsg(cls, text, turn) {
  const el = document.createElement("div");
  el.className = `msg ${cls}`;
  el.textContent = text;
  if (turn !== undefined) el.dataset.turn = turn;
  logEl.appendChild(el);
  el.scrollIntoView({ block: "end", behavior: "smooth" });
  return el;
}

function botMsg(turn) {
  const st = turnState(turn);
  if (!st.el) st.el = addMsg("bot", "…", turn);
  return st.el;
}

function turnState(turn) {
  if (!turns.has(turn)) {
    turns.set(turn, { sources: new Set(), firstScheduled: false, turnEndAt: null, audioEnded: false,
                      underruns: 0, el: null, latencyShown: false });
  }
  return turns.get(turn);
}

function send(obj) { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }

// ---- playback -------------------------------------------------------------
function playChunk(turn, pcm16, sampleRate) {
  if (cleared.has(turn)) return;
  const st = turnState(turn);
  const f32 = new Float32Array(pcm16.length);
  for (let i = 0; i < pcm16.length; i++) f32[i] = pcm16[i] / 0x8000;
  const buf = ctx.createBuffer(1, f32.length, sampleRate);
  buf.copyToChannel(f32, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);

  const now = ctx.currentTime;
  if (st.firstScheduled && nextPlayTime < now) st.underruns++; // gap in the audio stream
  const startAt = Math.max(now + 0.01, nextPlayTime);
  src.start(startAt);
  nextPlayTime = startAt + buf.duration;
  st.sources.add(src);
  src.onended = () => { st.sources.delete(src); maybePlaybackDone(turn); };

  if (!st.firstScheduled) {
    st.firstScheduled = true;
    const playAtPerf = performance.now() + (startAt - now) * 1000;
    const metric = { type: "client_metric", turn, first_chunk_lead_ms: Math.round((startAt - now) * 1000) };
    if (st.turnEndAt !== null) metric.turn_end_to_playback_ms = Math.round(playAtPerf - st.turnEndAt);
    send(metric);
    if (metric.turn_end_to_playback_ms !== undefined) showLatency(turn, metric.turn_end_to_playback_ms);
  }
}

function maybePlaybackDone(turn) {
  const st = turnState(turn);
  if (st.audioEnded && st.sources.size === 0 && !st.doneSent) {
    st.doneSent = true;
    send({ type: "client_metric", turn, underruns: st.underruns });
    send({ type: "playback_done", turn });
  }
}

function clearTurn(turn) {
  cleared.add(turn);
  const st = turns.get(turn);
  if (!st) return;
  for (const s of st.sources) { try { s.stop(); } catch (_) {} }
  st.sources.clear();
  nextPlayTime = 0;
  if (st.el) st.el.classList.add("cut");
}

function showLatency(turn, ms) {
  const el = botMsg(turn);
  const st = turnState(turn);
  if (st.latencyShown) return;
  st.latencyShown = true;
  const lat = document.createElement("span");
  lat.className = "lat";
  lat.textContent = `heard ${ms} ms after you stopped`;
  el.appendChild(lat);
}

// ---- server messages ------------------------------------------------------
let ttsRate = 24000;

function onServerMessage(m) {
  switch (m.type) {
    case "call_started": setStatus("Connecting…"); break;
    case "bot_audio_start": ttsRate = m.sample_rate || ttsRate; setStatus("Bot speaking"); botMsg(m.turn); break;
    case "bot_audio_end": { const st = turnState(m.turn); st.audioEnded = true; maybePlaybackDone(m.turn); setStatus("Listening"); break; }
    case "bot_text": { const el = botMsg(m.turn); el.firstChild.textContent = m.text; break; }
    case "user_turn_start": setStatus("You are speaking"); break;
    case "partial": partialEl.textContent = m.text; break;
    case "turn_end":
      partialEl.textContent = "";
      addMsg("user", m.text, m.turn);
      turnState(m.turn).turnEndAt = performance.now();
      setStatus("Thinking…");
      break;
    case "clear": clearTurn(m.turn); break;
    case "booking": showBooking(m.booking); break;
    case "error": addMsg("err", `${m.where}: ${m.message}`); break;
    case "call_summary": showSummary(m.summary); break;
  }
}

function showBooking(b) {
  const el = $("booking");
  el.hidden = false;
  el.innerHTML = "";
  const h = document.createElement("strong");
  h.textContent = `Booked ${b.booking_id}`;
  const p = document.createElement("div");
  p.textContent = `${b.patient_name} · ${b.doctor} · ${b.date} ${b.time} · ${b.phone}`;
  el.append(h, p);
}

function showSummary(s) {
  const lat = s.turns.map((t) => t.derived_ms.turn_end_to_first_audio_ms).filter((v) => v != null);
  const avg = lat.length ? Math.round(lat.reduce((a, b) => a + b, 0) / lat.length) : "–";
  const el = addMsg("err", "");
  el.style.color = "var(--muted)"; el.style.borderColor = "var(--line)";
  el.textContent = `Call ${s.call_id} ended: ${s.turns.length} turns, avg server voice-to-voice ${avg} ms. `;
  const a = document.createElement("a");
  a.href = `/dashboard#${s.call_id}`; a.textContent = "Open metrics";
  el.appendChild(a);
}

// ---- call control ---------------------------------------------------------
async function startCall() {
  callBtn.disabled = true;
  logEl.innerHTML = ""; $("booking").hidden = true; turns.clear(); cleared.clear();
  try {
    ctx = new AudioContext();
    await ctx.audioWorklet.addModule("/static/mic-worklet.js");
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (e) {
    addMsg("err", `Microphone: ${e.message}`); callBtn.disabled = false; return;
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    send({ type: "client_info", user_agent: navigator.userAgent, audio_context_rate: ctx.sampleRate,
           base_latency_ms: Math.round((ctx.baseLatency || 0) * 1000),
           output_latency_ms: Math.round((ctx.outputLatency || 0) * 1000) });
    srcNode = ctx.createMediaStreamSource(micStream);
    micNode = new AudioWorkletNode(ctx, "mic-processor");
    micNode.port.onmessage = (ev) => {
      if (ws.readyState === WebSocket.OPEN) ws.send(ev.data);
      const s = new Int16Array(ev.data);
      let peak = 0; for (let i = 0; i < s.length; i += 8) peak = Math.max(peak, Math.abs(s[i]));
      meterBar.style.width = `${Math.min(100, (peak / 0x7fff) * 220)}%`;
    };
    srcNode.connect(micNode);
    callBtn.textContent = "Hang up"; callBtn.className = "btn danger"; callBtn.disabled = false;
    callBtn.onclick = hangUp;
    setStatus("Connected");
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") { onServerMessage(JSON.parse(ev.data)); return; }
    const turn = new DataView(ev.data).getUint32(0, true);
    playChunk(turn, new Int16Array(ev.data, 4), ttsRate);
  };
  ws.onclose = () => teardown();
}

function hangUp() { send({ type: "hangup" }); callBtn.disabled = true; setStatus("Ending…"); }

function teardown() {
  for (const t of turns.keys()) clearTurn(t);
  if (micNode) micNode.disconnect();
  if (srcNode) srcNode.disconnect();
  if (micStream) micStream.getTracks().forEach((t) => t.stop());
  if (ctx) ctx.close();
  ws = null; ctx = null; micNode = null; srcNode = null; micStream = null;
  meterBar.style.width = "0";
  callBtn.textContent = "Start call"; callBtn.className = "btn primary"; callBtn.disabled = false;
  callBtn.onclick = startCall;
  setStatus("Idle");
}

callBtn.onclick = startCall;
