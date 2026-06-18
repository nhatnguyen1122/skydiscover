# OpenEvolve-Parity Signal Processing

Use this benchmark when comparing original SkyDiscover AdaEvolve or EvoX against:

- `FullReplacementMCTS/examples/signal_processing`
- `AdaEvolveMCTS/benchmarks/math/signal_processing_openevolve`
- `ShinkaEvolve/examples/openevolve_parity/signal_processing`

The initial program matches the OpenEvolve/FullReplacementMCTS signal-processing initial program. The evaluator uses the same scoring logic and returns both:

- `overall_score`: semantic primary metric for this benchmark
- `combined_score`: alias of `overall_score` for EvoX and other SkyDiscover code paths that assume this metric name

Candidates are evaluated through:

```python
run_signal_processing(noisy_signal=..., window_size=20)
```

The returned filtered signal should have length:

```python
len(noisy_signal) - window_size + 1
```

Do not mix this benchmark with `benchmarks/math/signal_processing`, which is the SkyDiscover-native signal-processing benchmark and uses a different scoring mechanism.

## Smoke

AdaEvolve:

```bash
conda run -n openevolve_test python -m skydiscover.cli \
  benchmarks/math/signal_processing_openevolve/initial_program.py \
  benchmarks/math/signal_processing_openevolve/evaluator.py \
  --config benchmarks/math/signal_processing_openevolve/config_adaevolve_nvidia.yaml \
  --output outputs/signal_processing_openevolve_adaevolve_smoke \
  --iterations 1
```

EvoX:

```bash
conda run -n openevolve_test python -m skydiscover.cli \
  benchmarks/math/signal_processing_openevolve/initial_program.py \
  benchmarks/math/signal_processing_openevolve/evaluator.py \
  --config benchmarks/math/signal_processing_openevolve/config_evox_nvidia.yaml \
  --output outputs/signal_processing_openevolve_evox_smoke \
  --iterations 1
```

## Full Runs

AdaEvolve:

```bash
conda run -n openevolve_test python -m skydiscover.cli \
  benchmarks/math/signal_processing_openevolve/initial_program.py \
  benchmarks/math/signal_processing_openevolve/evaluator.py \
  --config benchmarks/math/signal_processing_openevolve/config_adaevolve_nvidia.yaml \
  --output outputs/signal_processing_openevolve_adaevolve_full \
  --iterations 50
```

EvoX:

```bash
conda run -n openevolve_test python -m skydiscover.cli \
  benchmarks/math/signal_processing_openevolve/initial_program.py \
  benchmarks/math/signal_processing_openevolve/evaluator.py \
  --config benchmarks/math/signal_processing_openevolve/config_evox_nvidia.yaml \
  --output outputs/signal_processing_openevolve_evox_full \
  --iterations 50
```
