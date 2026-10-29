"""Print latency percentiles and per-call totals from logged calls.

    python -m scripts.report            # all calls
    python -m scripts.report --last 10  # most recent 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.metrics import LATENCY_KEYS, aggregate, load_summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="logs")
    ap.add_argument("--last", type=int, default=None)
    args = ap.parse_args()

    summaries = load_summaries(Path(args.dir), limit=args.last)
    if not summaries:
        print("No calls logged yet.")
        return
    agg = aggregate(summaries)
    print(f"{agg['calls']} calls, {agg['turns']} turns\n")
    print(f"{'metric':34} {'n':>4} {'p50':>8} {'p95':>8}")
    for key in LATENCY_KEYS:
        s = agg[key]
        p50 = "-" if s["p50"] is None else f"{s['p50']:.0f}"
        p95 = "-" if s["p95"] is None else f"{s['p95']:.0f}"
        print(f"{key:34} {s['n']:>4} {p50:>8} {p95:>8}")

    print(f"\n{'call':24} {'turns':>5} {'dur s':>6} {'tok in':>7} {'tok out':>7} {'chars':>6} {'spec h/m':>8} {'errs':>4}  booked")
    for s in summaries:
        t = s["totals"]
        print(f"{s['call_id']:24} {len(s['turns']):>5} {s['duration_ms']/1000:>6.1f} {t['input_tokens']:>7} "
              f"{t['output_tokens']:>7} {t['tts_chars']:>6} {t['speculative_hits']:>4}/{t['speculative_misses']:<3} "
              f"{t['turn_errors'] + len(s['errors']):>4}  {'yes' if s.get('outcome', {}).get('booking') else 'no'}")


if __name__ == "__main__":
    main()
