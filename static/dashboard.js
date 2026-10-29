const $ = (id) => document.getElementById(id);
const fmt = (v) => (v === null || v === undefined ? "–" : Math.round(v).toLocaleString());

const AGG_LABELS = {
  turn_end_to_first_audio_ms: "Voice-to-voice (server)",
  client_turn_end_to_playback_ms: "Voice-to-voice (browser)",
  llm_ttft_ms: "LLM first token",
  llm_first_phrase_ms: "LLM first phrase",
  llm_total_ms: "LLM total",
  tts_ttfa_ms: "TTS first audio",
  stt_eager_to_end_ms: "STT eager → end",
};

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "style") e.style.cssText = v; else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}

async function load() {
  const data = await (await fetch("/api/calls?limit=100")).json();
  renderAgg(data.aggregate);
  renderCalls(data.calls);
  const id = location.hash.slice(1) || (data.calls[0] && data.calls[0].call_id);
  if (id) showCall(id);
}

function renderAgg(a) {
  const box = $("agg");
  box.innerHTML = "";
  for (const [key, label] of Object.entries(AGG_LABELS)) {
    const s = a[key] || {};
    box.append(el("div", { class: "card" },
      el("div", { class: "k" }, label),
      el("div", { class: "v" }, fmt(s.p50), el("small", {}, ` p50 · p95 ${fmt(s.p95)} · n=${s.n || 0}`))));
  }
  box.append(el("div", { class: "card" }, el("div", { class: "k" }, "Calls / turns"),
    el("div", { class: "v" }, `${a.calls} / ${a.turns}`)));
}

function renderCalls(calls) {
  const t = $("calls");
  t.innerHTML = "";
  if (!calls.length) { t.append(el("tr", {}, el("td", { class: "empty" }, "No calls yet. Make one from the call page."))); return; }
  t.append(el("tr", {}, ...["Call", "Started", "Duration", "Turns", "Booked", "Tokens in/out", "TTS chars",
    "Spec hit/miss", "Barge-ins", "Errors", "End"].map((h) => el("th", {}, h))));
  for (const c of calls) {
    const x = c.totals;
    const row = el("tr", { class: "click", "data-id": c.call_id },
      el("td", {}, c.call_id), el("td", {}, new Date(c.started_at).toLocaleString()),
      el("td", {}, `${(c.duration_ms / 1000).toFixed(1)} s`), el("td", {}, c.turns),
      el("td", {}, c.booked ? "yes" : "no"), el("td", {}, `${x.input_tokens} / ${x.output_tokens}`),
      el("td", {}, x.tts_chars), el("td", {}, `${x.speculative_hits} / ${x.speculative_misses}`),
      el("td", {}, x.barge_ins), el("td", {}, x.turn_errors), el("td", {}, c.end_reason));
    row.onclick = () => { location.hash = c.call_id; showCall(c.call_id); };
    t.append(row);
  }
}

const SEGMENTS = [
  ["wait", "stt_turn_end", "llm_start", "--bar4", "Turn end → LLM start"],
  ["llm", "llm_start", "llm_first_token", "--bar1", "LLM to first token"],
  ["phrase", "llm_first_token", "llm_first_phrase", "--bar2", "First phrase"],
  ["tts", "tts_first_send", "tts_first_audio", "--bar3", "TTS to first audio"],
];

async function showCall(id) {
  document.querySelectorAll("tr.click").forEach((r) => r.classList.toggle("sel", r.dataset.id === id));
  const res = await fetch(`/api/calls/${encodeURIComponent(id)}`);
  const box = $("detail");
  box.innerHTML = "";
  if (!res.ok) { box.append(el("p", { class: "empty" }, `Call ${id} not found.`)); return; }
  const s = await res.json();
  box.append(el("h2", {}, `Call ${s.call_id}`));
  box.append(el("p", { class: "sub" },
    `STT connect ${fmt(s.derived_ms.stt_connect_ms)} ms · TTS connect ${fmt(s.derived_ms.tts_connect_ms)} ms · ` +
    `greeting audio at ${fmt(s.derived_ms.greeting_first_audio_ms)} ms · LLM ${s.config.llm_model} (${s.config.llm_effort}) · ` +
    `TTS ${s.config.tts_model} ${s.config.tts_language} speed ${s.config.tts_speed} · mic ${fmt(s.totals.mic_audio_ms / 1000)} s · ` +
    `bot audio ${fmt(s.totals.bot_audio_ms / 1000)} s`));
  if (s.outcome && s.outcome.booking) {
    const b = s.outcome.booking;
    box.append(el("p", {}, `Booked ${b.booking_id}: ${b.patient_name}, ${b.doctor}, ${b.date} ${b.time}`));
  }

  box.append(el("div", { class: "legend" }, ...SEGMENTS.map(([, , , c, label]) =>
    el("span", {}, el("b", { style: `background: var(${c})` }), label))));

  const spans = s.turns.map((t) => {
    const m = t.marks_ms;
    if (m.tts_first_audio === undefined) return null;
    const origin = Math.min(...["stt_turn_end", "llm_start", "tts_first_send"].map((k) => m[k]).filter((v) => v !== undefined));
    return { origin, end: m.tts_first_audio };
  });
  const scale = Math.max(1, ...spans.filter(Boolean).map((x) => x.end - x.origin));

  const t = el("table");
  t.append(el("tr", {}, ...["#", "User", "Bot", "Voice-to-voice", "Browser", "LLM TTFT", "TTS TTFA",
    "Waterfall", "Tokens", "Tools", "Spec", ""].map((h) => el("th", {}, h))));
  s.turns.forEach((turn, i) => {
    const d = turn.derived_ms, m = turn.marks_ms, span = spans[i];
    const wf = el("div", { class: "wf" });
    if (span) {
      for (const [, a, b, color] of SEGMENTS) {
        if (m[a] === undefined || m[b] === undefined || m[b] <= m[a]) continue;
        const left = ((m[a] - span.origin) / scale) * 100, width = ((m[b] - m[a]) / scale) * 100;
        wf.append(el("i", { style: `left:${left}%;width:${width}%;background:var(${color})`, title: `${a} → ${b}: ${Math.round(m[b] - m[a])} ms` }));
      }
    }
    const flags = [turn.interrupted ? "interrupted" : "", turn.errors.length ? `${turn.errors.length} error(s)` : ""].filter(Boolean).join(", ");
    t.append(el("tr", {},
      el("td", {}, turn.index), el("td", {}, turn.user_text || "—"), el("td", {}, turn.bot_text),
      el("td", {}, fmt(d.turn_end_to_first_audio_ms)), el("td", {}, fmt(d.client_turn_end_to_playback_ms)),
      el("td", {}, fmt(d.llm_ttft_ms)), el("td", {}, fmt(d.tts_ttfa_ms)), el("td", {}, wf),
      el("td", {}, `${turn.counts.input_tokens}/${turn.counts.output_tokens}`),
      el("td", {}, turn.tool_calls.map((c) => `${c.name}${c.ok ? "" : " ✗"}`).join(", ")),
      el("td", {}, turn.speculative), el("td", {}, flags)));
  });
  box.append(el("div", { class: "scroll" }, t));
  box.append(el("p", { class: "sub" }, el("a", { href: `/api/calls/${s.call_id}` }, "Raw summary JSON"),
    ` · full event log: logs/calls/${s.call_id}.events.jsonl`));
}

window.addEventListener("hashchange", () => { const id = location.hash.slice(1); if (id) showCall(id); });
load();
