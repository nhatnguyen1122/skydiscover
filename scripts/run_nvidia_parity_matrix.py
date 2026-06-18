#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_convergence import _read_json, _score_from_metrics, export_convergence


BENCHMARKS = {
    "circle_packing": {
        "alias": "cp",
        "initial": "benchmarks/math/circle_packing/initial_program.py",
        "evaluator": "benchmarks/math/circle_packing/evaluator.py",
        "config": "benchmarks/math/circle_packing/config_nvidia.yaml",
    },
    "circle_packing_n32": {
        "alias": "cp32",
        "initial": "benchmarks/math/circle_packing_n32/initial_program.py",
        "evaluator": "benchmarks/math/circle_packing_n32/evaluator.py",
        "config": "benchmarks/math/circle_packing_n32/config_nvidia.yaml",
    },
    "circle_packing_rect": {
        "alias": "cprect",
        "initial": "benchmarks/math/circle_packing_rect/initial_program.py",
        "evaluator": "benchmarks/math/circle_packing_rect/evaluator.py",
        "config": "benchmarks/math/circle_packing_rect/config_nvidia.yaml",
    },
    "erdos_min_overlap": {
        "alias": "erdos",
        "initial": "benchmarks/math/erdos_min_overlap/initial_program.py",
        "evaluator": "benchmarks/math/erdos_min_overlap/evaluator.py",
        "config": "benchmarks/math/erdos_min_overlap/config_nvidia.yaml",
    },
    "first_autocorr_ineq": {
        "alias": "auto1",
        "initial": "benchmarks/math/first_autocorr_ineq/initial_program.py",
        "evaluator": "benchmarks/math/first_autocorr_ineq/evaluator.py",
        "config": "benchmarks/math/first_autocorr_ineq/config_nvidia.yaml",
    },
    "heilbronn_convex/13": {
        "alias": "hconv13",
        "initial": "benchmarks/math/heilbronn_convex/13/initial_program.py",
        "evaluator": "benchmarks/math/heilbronn_convex/13/evaluator.py",
        "config": "benchmarks/math/heilbronn_convex/13/config_nvidia.yaml",
    },
    "heilbronn_convex/14": {
        "alias": "hconv14",
        "initial": "benchmarks/math/heilbronn_convex/14/initial_program.py",
        "evaluator": "benchmarks/math/heilbronn_convex/14/evaluator.py",
        "config": "benchmarks/math/heilbronn_convex/14/config_nvidia.yaml",
    },
    "heilbronn_triangle": {
        "alias": "heil",
        "initial": "benchmarks/math/heilbronn_triangle/initial_program.py",
        "evaluator": "benchmarks/math/heilbronn_triangle/evaluator.py",
        "config": "benchmarks/math/heilbronn_triangle/config_nvidia.yaml",
    },
    "hexagon_packing/11": {
        "alias": "hex11",
        "initial": "benchmarks/math/hexagon_packing/11/initial_program.py",
        "evaluator": "benchmarks/math/hexagon_packing/11/evaluator.py",
        "config": "benchmarks/math/hexagon_packing/11/config_nvidia.yaml",
    },
    "hexagon_packing/12": {
        "alias": "hex12",
        "initial": "benchmarks/math/hexagon_packing/12/initial_program.py",
        "evaluator": "benchmarks/math/hexagon_packing/12/evaluator.py",
        "config": "benchmarks/math/hexagon_packing/12/config_nvidia.yaml",
    },
    "matmul": {
        "alias": "matmul",
        "initial": "benchmarks/math/matmul/initial_program.py",
        "evaluator": "benchmarks/math/matmul/evaluator.py",
        "config": "benchmarks/math/matmul/config_nvidia.yaml",
    },
    "minimizing_max_min_dist/2": {
        "alias": "mmd2",
        "initial": "benchmarks/math/minimizing_max_min_dist/2/initial_program.py",
        "evaluator": "benchmarks/math/minimizing_max_min_dist/2/evaluator.py",
        "config": "benchmarks/math/minimizing_max_min_dist/2/config_nvidia.yaml",
    },
    "minimizing_max_min_dist/3": {
        "alias": "mmd3",
        "initial": "benchmarks/math/minimizing_max_min_dist/3/initial_program.py",
        "evaluator": "benchmarks/math/minimizing_max_min_dist/3/evaluator.py",
        "config": "benchmarks/math/minimizing_max_min_dist/3/config_nvidia.yaml",
    },
    "second_autocorr_ineq": {
        "alias": "auto2",
        "initial": "benchmarks/math/second_autocorr_ineq/initial_program.py",
        "evaluator": "benchmarks/math/second_autocorr_ineq/evaluator.py",
        "config": "benchmarks/math/second_autocorr_ineq/config_nvidia.yaml",
    },
    "sums_diffs_finite_sets": {
        "alias": "sumsdiffs",
        "initial": "benchmarks/math/sums_diffs_finite_sets/initial_program.py",
        "evaluator": "benchmarks/math/sums_diffs_finite_sets/evaluator.py",
        "config": "benchmarks/math/sums_diffs_finite_sets/config_nvidia.yaml",
    },
    "third_autocorr_ineq": {
        "alias": "auto3",
        "initial": "benchmarks/math/third_autocorr_ineq/initial_program.py",
        "evaluator": "benchmarks/math/third_autocorr_ineq/evaluator.py",
        "config": "benchmarks/math/third_autocorr_ineq/config_nvidia.yaml",
    },
    "uncertainty_ineq": {
        "alias": "uncert",
        "initial": "benchmarks/math/uncertainty_ineq/initial_program.py",
        "evaluator": "benchmarks/math/uncertainty_ineq/evaluator.py",
        "config": "benchmarks/math/uncertainty_ineq/config_nvidia.yaml",
    },
    "function_minimization": {
        "alias": "func",
        "initial": "benchmarks/openevolve_parity/function_minimization/initial_program.py",
        "evaluator": "benchmarks/openevolve_parity/function_minimization/evaluator.py",
        "config": "benchmarks/openevolve_parity/function_minimization/config_nvidia.yaml",
    },
    "k_module_problem": {
        "alias": "kmod",
        "initial": "benchmarks/openevolve_parity/k_module_problem/initial_program.py",
        "evaluator": "benchmarks/openevolve_parity/k_module_problem/evaluator.py",
        "config": "benchmarks/openevolve_parity/k_module_problem/config_nvidia.yaml",
    },
    "signal_processing": {
        "alias": "sp",
        "initial": "benchmarks/math/signal_processing/initial_program.py",
        "evaluator": "benchmarks/math/signal_processing/evaluator/evaluator.py",
        "config": "benchmarks/math/signal_processing/config_nvidia.yaml",
    },
    "signal_processing_openevolve": {
        "alias": "sp_oe",
        "initial": "benchmarks/math/signal_processing_openevolve/initial_program.py",
        "evaluator": "benchmarks/math/signal_processing_openevolve/evaluator.py",
        "config": "benchmarks/math/signal_processing_openevolve/config_adaevolve_nvidia.yaml",
        "strategy_configs": {
            "adaevolve": "benchmarks/math/signal_processing_openevolve/config_adaevolve_nvidia.yaml",
            "evox": "benchmarks/math/signal_processing_openevolve/config_evox_nvidia.yaml",
        },
    },
    "coral_trimul": {
        "alias": "trimul",
        "initial": "benchmarks/coral_trimul/initial_program.py",
        "evaluator": "benchmarks/coral_trimul/evaluator.py",
        "config": "benchmarks/coral_trimul/config_nvidia.yaml",
    },
    "coral_kernel_builder": {
        "alias": "kbuilder",
        "initial": "benchmarks/coral_kernel_builder/initial_program.py",
        "evaluator": "benchmarks/coral_kernel_builder/evaluator.py",
        "config": "benchmarks/coral_kernel_builder/config_nvidia.yaml",
    },
}

STRATEGIES = ("adaevolve", "evox")


def _parse_csv(value: str, choices: dict[str, Any] | tuple[str, ...]) -> list[str]:
    raw = [part.strip() for part in value.split(",") if part.strip()]
    aliases = {cfg["alias"]: name for name, cfg in BENCHMARKS.items()} if isinstance(choices, dict) else {}
    valid = set(choices) if not isinstance(choices, dict) else set(choices) | set(aliases)
    unknown = [item for item in raw if item not in valid]
    if unknown:
        raise ValueError(f"Unknown value(s): {', '.join(unknown)}")
    return [aliases.get(item, item) for item in raw]


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _best_from_convergence(path: Path) -> tuple[float | None, dict[str, Any]]:
    best_score = None
    best_row: dict[str, Any] = {}
    if not path.exists():
        return None, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        score = row.get("score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            if best_score is None or float(score) > best_score:
                best_score = float(score)
                best_row = row
    return best_score, best_row


def _best_from_run(run_dir: Path, convergence_path: Path) -> tuple[float | None, dict[str, Any]]:
    """Return score and metrics for the actual best program, not the final iteration."""
    best_info = _read_json(run_dir / "best" / "best_program_info.json")
    metrics = best_info.get("metrics") or {}
    _, score = _score_from_metrics(metrics)
    if score is not None:
        return score, {
            "event": "best",
            "program_id": best_info.get("id"),
            "parent_id": best_info.get("parent_id"),
            "iteration": best_info.get("iteration") or best_info.get("generation"),
            "metrics": metrics,
            "source": "best/best_program_info.json",
        }
    return _best_from_convergence(convergence_path)


def run_one(
    benchmark: str,
    strategy: str,
    run_idx: int,
    output_root: Path,
    iterations: int | None,
    resume: bool,
) -> dict[str, Any]:
    cfg = BENCHMARKS[benchmark]
    run_dir = output_root / benchmark / strategy / f"run_{run_idx:02d}"
    convergence_path = run_dir / "convergence.jsonl"
    if resume and convergence_path.exists():
        best_score, best_row = _best_from_run(run_dir, convergence_path)
        return {
            "benchmark": benchmark,
            "strategy": strategy,
            "run": run_idx,
            "status": "skipped",
            "output_dir": str(run_dir),
            "primary_score": best_score,
            "metrics": best_row.get("metrics", {}),
        }

    command = [
        sys.executable,
        "-m",
        "skydiscover.cli",
        cfg["initial"],
        cfg["evaluator"],
        "--config",
        cfg.get("strategy_configs", {}).get(strategy, cfg["config"]),
        "--search",
        strategy,
        "--output",
        str(run_dir),
    ]
    if iterations is not None:
        command.extend(["--iterations", str(iterations)])

    start = time.time()
    proc = subprocess.run(command, text=True)
    elapsed = time.time() - start

    export_convergence(run_dir, convergence_path)
    best_score, best_row = _best_from_run(run_dir, convergence_path)
    return {
        "benchmark": benchmark,
        "strategy": strategy,
        "run": run_idx,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "output_dir": str(run_dir),
        "convergence_path": str(convergence_path),
        "primary_score": best_score,
        "metrics": best_row.get("metrics", {}),
        "elapsed_seconds": elapsed,
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkyDiscover NVIDIA parity matrix")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--benchmarks", default="circle_packing,function_minimization,k_module_problem,signal_processing")
    parser.add_argument("--strategies", default="adaevolve,evox")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/nvidia_parity_matrix")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if not os.environ.get("NVIDIA_API_KEY"):
        parser.error("NVIDIA_API_KEY is not set")

    benchmarks = _parse_csv(args.benchmarks, BENCHMARKS)
    strategies = _parse_csv(args.strategies, STRATEGIES)
    output_root = Path(args.output_root)
    results_path = output_root / "results.jsonl"

    for benchmark in benchmarks:
        for strategy in strategies:
            for run_idx in range(1, args.runs + 1):
                print(f"[{benchmark} / {strategy} / run {run_idx}]")
                record = run_one(
                    benchmark=benchmark,
                    strategy=strategy,
                    run_idx=run_idx,
                    output_root=output_root,
                    iterations=args.iterations,
                    resume=args.resume,
                )
                _append_jsonl(results_path, record)
                if args.fail_fast and record["status"] == "failed":
                    raise SystemExit(record.get("returncode") or 1)

    print(f"Wrote matrix results to {results_path}")


if __name__ == "__main__":
    main()
