#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCORE_KEYS = ("combined_score", "overall_score", "composite_score", "score", "fitness", "accuracy")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _score_from_metrics(metrics: dict[str, Any] | None) -> tuple[str | None, float | None]:
    if not metrics:
        return None, None
    for key in SCORE_KEYS:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return key, float(value)
    numeric = [v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric:
        return "numeric_average", float(sum(numeric) / len(numeric))
    return None, None


def _append_program_row(rows: list[dict[str, Any]], program: dict[str, Any], source: str) -> None:
    metrics = program.get("metrics") or {}
    primary_metric, score = _score_from_metrics(metrics)
    if score is None:
        return
    rows.append(
        {
            "iteration": int(program.get("iteration_found") or program.get("generation") or 0),
            "event": "program",
            "program_id": program.get("id"),
            "parent_id": program.get("parent_id"),
            "score": score,
            "primary_metric": primary_metric,
            "metrics": metrics,
            "timestamp": program.get("timestamp"),
            "source": source,
        }
    )


def _rows_from_adaevolve_stats(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("adaevolve_iteration_stats_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            result = data.get("iteration_result") or {}
            child = result.get("child_program") or {}
            metrics = child.get("metrics") or {}
            primary_metric, score = _score_from_metrics(metrics)
            best_score = data.get("global_best_score")
            if best_score is None:
                best_metrics = ((data.get("best_program") or {}).get("metrics") or {})
                _, best_score = _score_from_metrics(best_metrics)
            rows.append(
                {
                    "iteration": int(data.get("iteration") or child.get("generation") or 0),
                    "event": "iteration",
                    "program_id": child.get("id"),
                    "parent_id": child.get("parent_id"),
                    "score": score,
                    "best_score": best_score,
                    "primary_metric": primary_metric,
                    "metrics": metrics,
                    "iteration_time": result.get("iteration_time_seconds"),
                    "llm_generation_time": result.get("llm_generation_time_seconds"),
                    "eval_time": result.get("eval_time_seconds"),
                    "error": result.get("error"),
                    "timestamp": data.get("timestamp"),
                    "source": str(path.relative_to(run_dir)),
                }
            )
    return rows


def _rows_from_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in sorted((run_dir / "checkpoints").glob("checkpoint_*")):
        programs_dir = checkpoint / "programs"
        if not programs_dir.exists():
            continue
        for program_path in sorted(programs_dir.glob("*.json")):
            _append_program_row(rows, _read_json(program_path), str(program_path.relative_to(run_dir)))
    return rows


def _rows_from_best(run_dir: Path) -> list[dict[str, Any]]:
    info = _read_json(run_dir / "best" / "best_program_info.json")
    metrics = info.get("metrics") or {}
    primary_metric, score = _score_from_metrics(metrics)
    if score is None:
        return []
    return [
        {
            "iteration": int(info.get("iteration") or info.get("generation") or 0),
            "event": "best",
            "program_id": info.get("id"),
            "parent_id": info.get("parent_id"),
            "score": score,
            "primary_metric": primary_metric,
            "metrics": metrics,
            "timestamp": info.get("timestamp"),
            "source": "best/best_program_info.json",
        }
    ]


def export_convergence(run_dir: Path, output_path: Path) -> int:
    rows = _rows_from_adaevolve_stats(run_dir)
    if not rows:
        rows = _rows_from_checkpoints(run_dir)
    if not rows:
        rows = _rows_from_best(run_dir)

    rows.sort(key=lambda row: (int(row.get("iteration") or 0), row.get("timestamp") or 0, row.get("program_id") or ""))
    best_score: float | None = None
    seen: set[tuple[Any, Any, Any]] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("iteration"), row.get("program_id"), row.get("event"))
        if key in seen:
            continue
        seen.add(key)
        score = row.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            best_score = float(score) if best_score is None else max(best_score, float(score))
        if row.get("best_score") is None:
            row["best_score"] = best_score
        normalized.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(row, default=str) + "\n")
    return len(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SkyDiscover run convergence.jsonl")
    parser.add_argument("run_dir", help="SkyDiscover output run directory")
    parser.add_argument("-o", "--output", default=None, help="Output JSONL path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "convergence.jsonl"
    count = export_convergence(run_dir, output)
    print(f"Wrote {count} convergence rows to {output}")


if __name__ == "__main__":
    main()
