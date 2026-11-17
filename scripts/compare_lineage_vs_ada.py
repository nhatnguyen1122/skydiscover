#!/usr/bin/env python3
"""Compare AdaEvolve vs LineageEvolve runs: score, runtime, token usage.

Usage: python scripts/compare_lineage_vs_ada.py outputs/lineage_vs_ada [--out report.md]
Expects <root>/<benchmark>/<method>/run_XX/ as written by scripts/run_lineage_matrix.sh.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

ALIAS = {"circle_packing": "cp26", "circle_packing_n32": "cp32", "circle_packing_rect": "cprect"}
MARKS = (25, 50, 75, 100)


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_run(run: Path) -> dict:
    summary = read_json(run / "run_summary.json")
    best = read_json(run / "best" / "best_program_info.json")
    metrics = best.get("metrics") or summary.get("best_metrics") or {}
    rows = []
    for f in sorted(run.glob("adaevolve_iteration_stats_*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    curve, statuses, best_so_far, last_it = {}, Counter(), None, 0
    for r in rows:
        it = int(r.get("iteration") or 0)
        last_it = max(last_it, it)
        res = r.get("iteration_result") or {}
        child = (res.get("child_program") or {}).get("metrics") or {}
        s = child.get("combined_score")
        if isinstance(s, (int, float)) and (best_so_far is None or s > best_so_far):
            best_so_far = s
        g = r.get("global_best_score")
        cand = [x for x in (g, best_so_far) if isinstance(x, (int, float))]
        curve[it] = max(cand) if cand else None
        err = res.get("error")
        if err is None:
            statuses["admitted"] += 1
        else:
            statuses[str(err).split(":")[0].strip()[:40]] += 1
    usage = (summary.get("token_usage") or {}).get("total") or {}
    by_purpose = (summary.get("token_usage") or {}).get("by_purpose") or {}
    wall = summary.get("wall_clock_seconds")
    if not summary:
        # Run stopped early: take cumulative usage / elapsed time from the
        # last line of the per-iteration logs.
        for pattern in ("lineageevolve_iterations_*.jsonl", "adaevolve_iteration_stats_*.jsonl"):
            files = sorted(run.glob(pattern))
            if not files:
                continue
            lines = [ln for ln in files[-1].read_text().splitlines() if ln.strip()]
            if lines:
                last = json.loads(lines[-1])
                usage = last.get("cumulative_token_usage") or usage
                wall = last.get("elapsed_seconds") or wall
                break
    at = {}
    for m in MARKS:
        vals = [v for k, v in curve.items() if k <= m and v is not None]
        at[m] = max(vals) if vals else None
    return {
        "run": run.name,
        "finished": bool(summary),
        "iterations_logged": last_it,
        "score": max(
            [v for v in [metrics.get("combined_score")] + list(curve.values()) if isinstance(v, (int, float))],
            default=None,
        ),
        "test_score": metrics.get("test_combined_score"),
        "best_iteration": best.get("iteration"),
        "wall_clock_s": wall,
        "llm_calls": usage.get("calls"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "tokens_by_purpose": {k: v.get("total_tokens") for k, v in by_purpose.items()},
        "calls_by_purpose": {k: v.get("calls") for k, v in by_purpose.items()},
        "statuses": dict(statuses),
        "best_at": at,
    }


def fmt(x, nd=4):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.mean(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    data = {}
    for bdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for mdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            runs = [load_run(r) for r in sorted(mdir.glob("run_*"))]
            if runs:
                data.setdefault(bdir.name, {})[mdir.name] = runs

    L = ["# AdaEvolve vs LineageEvolve", "", "## Per run", "",
         "| Benchmark | Method | Run | Done | Best score | @25 | @50 | @75 | @100 | Best iter | Wall clock (min) | LLM calls | Prompt tok | Completion tok | Total tok |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, methods in data.items():
        for m, runs in methods.items():
            for r in runs:
                wc = r["wall_clock_s"] / 60 if r["wall_clock_s"] else None
                L.append(
                    f"| {ALIAS.get(b, b)} | {m} | {r['run']} | {'yes' if r['finished'] else f'no ({r['iterations_logged']})'} | "
                    f"{fmt(r['score'], 6)} | " + " | ".join(fmt(r['best_at'][k]) for k in MARKS)
                    + f" | {fmt(r['best_iteration'])} | {fmt(wc, 1)} | {fmt(r['llm_calls'])} | "
                    f"{fmt(r['prompt_tokens'])} | {fmt(r['completion_tokens'])} | {fmt(r['total_tokens'])} |"
                )
    L += ["", "## Mean over runs", "",
          "| Benchmark | Method | Runs | Mean best | Min best | Max best | Mean wall clock (min) | Mean LLM calls | Mean total tokens |",
          "|---|---|---|---|---|---|---|---|---|"]
    for b, methods in data.items():
        for m, runs in methods.items():
            scores = [r["score"] for r in runs if isinstance(r["score"], (int, float))]
            wc = mean([r["wall_clock_s"] for r in runs])
            L.append(
                f"| {ALIAS.get(b, b)} | {m} | {len(runs)} | {fmt(mean(scores), 6)} | "
                f"{fmt(min(scores) if scores else None, 6)} | {fmt(max(scores) if scores else None, 6)} | {fmt(wc / 60 if wc else None, 1)} | "
                f"{fmt(mean([r['llm_calls'] for r in runs]), 0)} | {fmt(mean([r['total_tokens'] for r in runs]), 0)} |"
            )
    L += ["", "## Iteration outcomes and token split", "",
          "| Benchmark | Method | Run | Outcomes | Tokens by purpose | Calls by purpose |", "|---|---|---|---|---|---|"]
    for b, methods in data.items():
        for m, runs in methods.items():
            for r in runs:
                L.append(f"| {ALIAS.get(b, b)} | {m} | {r['run']} | {r['statuses']} | {r['tokens_by_purpose']} | {r['calls_by_purpose']} |")
    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        Path(args.out).with_suffix(".json").write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
