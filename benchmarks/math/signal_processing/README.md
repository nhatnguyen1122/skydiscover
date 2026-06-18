# Native SkyDiscover Signal Processing

Evolve an adaptive filtering algorithm for non-stationary time series data using SkyDiscover's native signal-processing evaluator. This benchmark is separate from `signal_processing_openevolve`, which is the OpenEvolve-parity signal-processing benchmark.

## Problem

**Input**: Univariate time series with non-linear dynamics, non-stationary statistics, and rapidly changing spectral characteristics.

**Evaluator call**: `run_signal_processing(noisy_signal=..., window_size=20)`.

**Output contract**: return a filtered signal with length `len(noisy_signal) - window_size + 1`.

**Multi-objective function**:
```
J(theta) = 0.3*S + 0.2*L_recent + 0.2*L_avg + 0.3*R
```
- **S**: Slope change penalty (directional reversals in filtered signal)
- **L_recent**: Instantaneous lag error
- **L_avg**: Average tracking error
- **R**: False reversal penalty (noise-induced trend changes)

The evaluator tests on 5 synthetic signals: sinusoidal, multi-frequency, non-stationary, step changes, and random walk.

## Run

```bash
# From repo root
uv run skydiscover-run \
  benchmarks/math/signal_processing/initial_program.py \
  benchmarks/math/signal_processing/evaluator/evaluator.py \
  -c benchmarks/math/signal_processing/config.yaml \
  -s [your_algorithm] \
  -i 100
```

NVIDIA matrix runner alias:

```bash
python scripts/run_nvidia_parity_matrix.py \
  --benchmarks sp \
  --strategies adaevolve \
  --iterations 300 \
  --output-root outputs/native_sp_adaevolve
```

Use `sp` for this native SkyDiscover benchmark and `sp_oe` for the OpenEvolve-parity signal-processing benchmark.

## Scoring

- **combined_score**: Composite J(theta) metric (higher is better)
- Also reports: slope changes, correlation, lag error, noise reduction, processing time

## Files

| File | Description |
|------|-------------|
| `initial_program.py` | Seed: basic moving average / weighted exponential filters |
| `evaluator.py` | Multi-objective evaluation across 5 synthetic test signals |
| `config.yaml` | LLM and evaluator settings |
| `requirements.txt` | Python dependencies |
