from __future__ import annotations

import importlib.util
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


PROBLEMS = {
    "oscillator1": {"columns": ["x", "v"], "target": "a"},
    "oscillator2": {"columns": ["t", "x", "v"], "target": "a"},
    "bactgrow": {"columns": ["b", "s", "temp", "pH"], "target": "db"},
    "stressstrain": {"columns": ["strain", "temp"], "target": "stress"},
}


def _load_candidate(program_path: str):
    spec = importlib.util.spec_from_file_location("llmsr_candidate", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate from {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "LLMSR"):
        raise AttributeError("Candidate must define class LLMSR")
    candidate = module.LLMSR()
    if not hasattr(candidate, "equation"):
        raise AttributeError("LLMSR must define equation(self, inputs, params)")
    return candidate


def _synthetic_split(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    inputs = np.column_stack(
        [
            rng.uniform(0.0, 8.0, n),
            rng.uniform(-1.5, 1.5, n),
            rng.uniform(-1.0, 1.0, n),
        ]
    )
    t, x, v = inputs[:, 0], inputs[:, 1], inputs[:, 2]
    outputs = 1.7 * x - 0.55 * x**3 + 0.08 * x**5 - 0.35 * v + 0.03 * v**3 + 0.25 * np.sin(t)
    return {"inputs": inputs.astype(np.float64), "outputs": outputs.astype(np.float64)}


def _load_synthetic_problem() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(20260620)
    return _synthetic_split(rng, 256), _synthetic_split(rng, 192), _synthetic_split(rng, 192)


def _load_llmsr_csv_problem(problem_name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    data_root = os.environ.get("LLM_SR_DATA_ROOT")
    data_path = os.environ.get("LLM_SR_DATA_PATH")
    if data_path is None:
        if data_root is None:
            for parent in Path(__file__).resolve().parents:
                candidate = parent / "LLM-SR" / "data"
                if candidate.exists():
                    data_root = str(candidate)
                    break
        if data_root is None:
            raise FileNotFoundError(
                "Could not find LLM-SR/data. Set LLM_SR_DATA_ROOT to the LLM-SR data directory."
            )
        data_path = os.path.join(data_root, problem_name)

    spec = PROBLEMS[problem_name]
    columns = spec["columns"]
    target = spec["target"]

    import pandas as pd

    def read_csv_split(filename: str) -> dict[str, np.ndarray]:
        path = os.path.join(data_path, filename)
        frame = pd.read_csv(path)
        missing = set(columns + [target]) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return {
            "inputs": frame[columns].to_numpy(dtype=np.float64),
            "outputs": frame[target].to_numpy(dtype=np.float64),
        }

    return read_csv_split("train.csv"), read_csv_split("test_id.csv"), read_csv_split("test_ood.csv")


def _load_problem() -> tuple[str, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if os.environ.get("PACE_LLMSR_SYNTHETIC") == "1":
        train, test, ood = _load_synthetic_problem()
        return "synthetic_oscillator2", train, test, ood

    problem_name = os.environ.get("LLM_SR_PROBLEM_NAME", "oscillator2")
    if problem_name not in PROBLEMS:
        raise ValueError(f"Unknown LLM_SR_PROBLEM_NAME={problem_name!r}; valid: {sorted(PROBLEMS)}")
    train, test, ood = _load_llmsr_csv_problem(problem_name)
    return problem_name, train, test, ood


def _call_equation(candidate: Any, inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
    y_pred = candidate.equation(inputs, params)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_pred.shape != (inputs.shape[0],):
        y_pred = y_pred.reshape(-1)
    if y_pred.shape != (inputs.shape[0],):
        raise ValueError(f"equation returned shape {y_pred.shape}, expected {(inputs.shape[0],)}")
    return y_pred


def _base_metrics(y_pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(y_pred)
    if not np.any(finite):
        return {"mse": float("inf"), "nmse": float("inf"), "log10_nmse": float("inf")}
    y_pred = y_pred[finite]
    y = y[finite]
    mse = float(np.mean((y - y_pred) ** 2))
    var = float(np.var(y))
    nmse = mse / max(var, 1e-12)
    return {"mse": mse, "nmse": nmse, "log10_nmse": float(np.log10(max(nmse, 1e-300)))}


def _fit_and_score(candidate: Any, train: dict[str, np.ndarray], test: dict[str, np.ndarray], ood: dict[str, np.ndarray]) -> dict[str, float]:
    n_params = int(os.environ.get("LLM_SR_NUM_PARAMS", "12"))
    x_train = train["inputs"]
    y_train = train["outputs"]

    def loss(params: np.ndarray) -> float:
        try:
            pred = _call_equation(candidate, x_train, params)
            if not np.all(np.isfinite(pred)):
                return 1e100
            return float(np.mean((pred - y_train) ** 2))
        except Exception:
            return 1e100

    starts = [np.ones(n_params), np.zeros(n_params)]
    best = None
    for start in starts:
        result = minimize(loss, start, method="BFGS")
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    params = best.x

    test_metrics = _base_metrics(_call_equation(candidate, test["inputs"], params), test["outputs"])
    ood_metrics = _base_metrics(_call_equation(candidate, ood["inputs"], params), ood["outputs"])
    combined_score = -0.7 * test_metrics["log10_nmse"] - 0.3 * ood_metrics["log10_nmse"]
    return {
        "combined_score": float(combined_score),
        "log10_nmse": test_metrics["log10_nmse"],
        "ood_log10_nmse": ood_metrics["log10_nmse"],
        "mse": test_metrics["mse"],
        "ood_mse": ood_metrics["mse"],
        "correct": 1.0,
    }


def evaluate(program_path: str) -> dict[str, Any]:
    try:
        problem_name, train, test, ood = _load_problem()
        candidate = _load_candidate(program_path)
        metrics = _fit_and_score(candidate, train, test, ood)
        metrics["problem_name"] = problem_name
        return metrics
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "correct": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
