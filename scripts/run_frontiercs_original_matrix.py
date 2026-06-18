#!/usr/bin/env python3
"""Run Frontier-CS problems with original SkyDiscover AdaEvolve/EvoX."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "benchmarks" / "frontier-cs-eval"
FRONTIER_DIR = BENCH_DIR / "Frontier-CS"
STRATEGIES = {"adaevolve", "evox"}


def available_problems() -> list[str]:
    problems = FRONTIER_DIR / "algorithmic" / "problems"
    if not problems.exists():
        return []
    return sorted((p.name for p in problems.iterdir() if p.is_dir() and p.name.isdigit()), key=int)


def parse_csv(value: str, valid: set[str] | None = None) -> list[str]:
    if value == "all" and valid is None:
        problems = available_problems()
        if not problems:
            raise ValueError("No Frontier-CS problems found. Clone Frontier-CS first.")
        return problems
    items = [x.strip() for x in value.split(",") if x.strip()]
    if valid is not None:
        unknown = sorted(set(items) - valid)
        if unknown:
            raise ValueError(f"Unknown value(s): {', '.join(unknown)}")
    return items


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_last_jsonl(path: Path) -> dict[str, Any]:
    last: dict[str, Any] = {}
    if not path.exists():
        return last
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


def problem_prompt(problem_id: str) -> str:
    statement = FRONTIER_DIR / "algorithmic" / "problems" / problem_id / "statement.txt"
    config = FRONTIER_DIR / "algorithmic" / "problems" / problem_id / "config.yaml"
    statement_text = statement.read_text(encoding="utf-8") if statement.exists() else f"Frontier-CS problem {problem_id}"
    config_text = config.read_text(encoding="utf-8") if config.exists() else ""
    return f"""You are an expert competitive programmer specializing in algorithmic optimization.

FRONTIER-CS PROBLEM ID: {problem_id}

PROBLEM STATEMENT:
{statement_text}

CONSTRAINTS AND JUDGE CONFIG:
{config_text}

OBJECTIVE: Maximize the score returned by the Frontier-CS judge. Higher is better.
Your solution must be valid C++ code with main(), reading from stdin and writing to stdout.
Return complete C++ code only.
"""


def prepare_config(problem_id: str, strategy: str, output_root: Path, iterations: int) -> str:
    with (BENCH_DIR / "config.yaml").open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["language"] = "cpp"
    data["max_iterations"] = iterations
    data["checkpoint_interval"] = min(25, max(1, iterations)) if iterations else 1
    data["diff_based_generation"] = bool(data.get("diff_based_generation", True))
    data["max_solution_length"] = int(data.get("max_solution_length", 60000))
    data["random_seed"] = 42
    data["prompt"] = {"system_message": problem_prompt(problem_id)}
    data["llm"] = {
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key": "${NVIDIA_API_KEY}",
        "models": [{"name": "openai/gpt-oss-120b", "api_base": "https://integrate.api.nvidia.com/v1", "api_key": "${NVIDIA_API_KEY}", "weight": 1.0}],
        "evaluator_models": [{"name": "openai/gpt-oss-120b", "api_base": "https://integrate.api.nvidia.com/v1", "api_key": "${NVIDIA_API_KEY}", "weight": 1.0}],
        "guide_models": [{"name": "openai/gpt-oss-120b", "api_base": "https://integrate.api.nvidia.com/v1", "api_key": "${NVIDIA_API_KEY}", "weight": 1.0}],
        "temperature": 0.2,
        "max_tokens": 4096,
        "timeout": 600,
        "retries": 3,
        "retry_delay": 5,
    }
    data["search"] = {"type": strategy}
    data.setdefault("evaluator", {})["cascade_evaluation"] = False
    cfg_dir = output_root / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"problem_{problem_id}_{strategy}.yaml"
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(cfg_path.relative_to(ROOT))


def summarize(problem_id: str, strategy: str, run: int, run_dir: Path, returncode: int, elapsed: float) -> dict[str, Any]:
    best = read_json(run_dir / "best" / "best_program_info.json")
    conv = read_last_jsonl(run_dir / "convergence.jsonl")
    metrics = best.get("metrics") or conv.get("metrics") or {}
    return {
        "problem": problem_id,
        "strategy": strategy,
        "run": run,
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "output_dir": str(run_dir),
        "elapsed_seconds": round(elapsed, 3),
        "primary_score": metrics.get("combined_score", conv.get("best_score")),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default="0", help="Comma list of problem IDs, or 'all'.")
    parser.add_argument("--strategies", default="adaevolve,evox")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-root", default="outputs/frontiercs_original")
    parser.add_argument("--judge-urls", default=os.environ.get("JUDGE_URLS", "http://localhost:8081"))
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    problems = parse_csv(args.problems)
    strategies = parse_csv(args.strategies, STRATEGIES)
    if not args.dry_run and not os.environ.get("NVIDIA_API_KEY"):
        parser.error("NVIDIA_API_KEY is not set")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for problem_id in problems:
        for strategy in strategies:
            cfg = prepare_config(problem_id, strategy, output_root, args.iterations)
            for run_idx in range(1, args.runs + 1):
                run_dir = output_root / f"problem_{problem_id}" / strategy / f"run_{run_idx:02d}"
                if args.skip_complete and (run_dir / "best" / "best_program_info.json").exists():
                    row = summarize(problem_id, strategy, run_idx, run_dir, 0, 0.0)
                    row["status"] = "skipped"
                    rows.append(row)
                    continue
                cmd = [
                    sys.executable,
                    "-m",
                    "skydiscover.cli",
                    "benchmarks/frontier-cs-eval/initial_program.cpp",
                    "benchmarks/frontier-cs-eval/evaluator.py",
                    "--config",
                    cfg,
                    "--search",
                    strategy,
                    "--output",
                    str(run_dir),
                    "--iterations",
                    str(args.iterations),
                ]
                print("+ " + " ".join(cmd))
                if args.dry_run:
                    continue
                env = os.environ.copy()
                env["FRONTIER_CS_PROBLEM"] = str(problem_id)
                env["JUDGE_URLS"] = args.judge_urls
                run_dir.mkdir(parents=True, exist_ok=True)
                start = time.time()
                with (run_dir / "run.log").open("w", encoding="utf-8") as log:
                    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                row = summarize(problem_id, strategy, run_idx, run_dir, proc.returncode, time.time() - start)
                rows.append(row)
                with (output_root / "results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                print(f"[problem {problem_id} {strategy} run {run_idx}] {row['status']} score={row['primary_score']}")
                if args.fail_fast and proc.returncode != 0:
                    return proc.returncode or 1
    if rows:
        keys = sorted({k for row in rows for k in row if k != "metrics"})
        with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows([{k: v for k, v in row.items() if k != "metrics"} for row in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
