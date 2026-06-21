#!/usr/bin/env python3
"""Run all four LLM-SR symbolic-regression problems with skydiscover."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBLEMS = ("oscillator1", "oscillator2", "bactgrow", "stressstrain")


def parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - set(PROBLEMS))
    if unknown:
        raise ValueError(f"Unknown LLM-SR problem(s): {', '.join(unknown)}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", default=",".join(PROBLEMS))
    parser.add_argument("--strategies", default="adaevolve")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--output-root", default="outputs/llmsr_symbolic_regression")
    parser.add_argument("--data-root", default="../LLM-SR/data")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    problems = parse_csv(args.problems)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    for problem in problems:
        env = os.environ.copy()
        env.setdefault("LLM_SR_DATA_ROOT", args.data_root)
        env["LLM_SR_PROBLEM_NAME"] = problem
        cmd = [
            sys.executable,
            "scripts/run_nvidia_parity_matrix.py",
            "--runs",
            str(args.runs),
            "--benchmarks",
            "pace_symreg",
            "--strategies",
            args.strategies,
            "--iterations",
            str(args.iterations),
            "--output-root",
            str(output_root / problem),
        ]
        if args.resume:
            cmd.append("--resume")
        if args.fail_fast:
            cmd.append("--fail-fast")
        print("+ " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        if args.fail_fast and proc.returncode != 0:
            return proc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
