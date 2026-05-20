# NVIDIA GPT-OSS-120B Parity Runs

This setup runs SkyDiscover AdaEvolve and EvoX on the same four benchmark tasks used for the FullReplacementMCTS comparison:

- circle packing
- function minimization
- k-module problem
- signal processing

Set the NVIDIA NIM key in the shell before running. Do not commit the key.

```bash
export NVIDIA_API_KEY="<your-nvidia-nim-key>"
```

Run commands from the `skydiscover` repository root. If the package is not already installed in the environment, install it first:

```bash
conda run -n openevolve_test pip install -e ".[math]"
```

## Smoke Tests

Run one iteration per benchmark and strategy:

```bash
conda run -n openevolve_test python -m skydiscover.cli \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config_nvidia.yaml \
  --search adaevolve \
  --iterations 1 \
  --output outputs/nvidia_smoke/circle_packing_adaevolve

conda run -n openevolve_test python -m skydiscover.cli \
  benchmarks/math/circle_packing/initial_program.py \
  benchmarks/math/circle_packing/evaluator.py \
  --config benchmarks/math/circle_packing/config_nvidia.yaml \
  --search evox \
  --iterations 1 \
  --output outputs/nvidia_smoke/circle_packing_evox
```

Use the same pattern for the other benchmarks:

```bash
# function minimization
conda run -n openevolve_test python -m skydiscover.cli benchmarks/openevolve_parity/function_minimization/initial_program.py benchmarks/openevolve_parity/function_minimization/evaluator.py --config benchmarks/openevolve_parity/function_minimization/config_nvidia.yaml --search adaevolve --iterations 1 --output outputs/nvidia_smoke/function_minimization_adaevolve
conda run -n openevolve_test python -m skydiscover.cli benchmarks/openevolve_parity/function_minimization/initial_program.py benchmarks/openevolve_parity/function_minimization/evaluator.py --config benchmarks/openevolve_parity/function_minimization/config_nvidia.yaml --search evox --iterations 1 --output outputs/nvidia_smoke/function_minimization_evox

# k-module problem
conda run -n openevolve_test python -m skydiscover.cli benchmarks/openevolve_parity/k_module_problem/initial_program.py benchmarks/openevolve_parity/k_module_problem/evaluator.py --config benchmarks/openevolve_parity/k_module_problem/config_nvidia.yaml --search adaevolve --iterations 1 --output outputs/nvidia_smoke/k_module_problem_adaevolve
conda run -n openevolve_test python -m skydiscover.cli benchmarks/openevolve_parity/k_module_problem/initial_program.py benchmarks/openevolve_parity/k_module_problem/evaluator.py --config benchmarks/openevolve_parity/k_module_problem/config_nvidia.yaml --search evox --iterations 1 --output outputs/nvidia_smoke/k_module_problem_evox

# signal processing
conda run -n openevolve_test python -m skydiscover.cli benchmarks/math/signal_processing/initial_program.py benchmarks/math/signal_processing/evaluator/evaluator.py --config benchmarks/math/signal_processing/config_nvidia.yaml --search adaevolve --iterations 1 --output outputs/nvidia_smoke/signal_processing_adaevolve
conda run -n openevolve_test python -m skydiscover.cli benchmarks/math/signal_processing/initial_program.py benchmarks/math/signal_processing/evaluator/evaluator.py --config benchmarks/math/signal_processing/config_nvidia.yaml --search evox --iterations 1 --output outputs/nvidia_smoke/signal_processing_evox
```

## Full Runs

Omit `--iterations 1` to use the benchmark config budgets:

- circle packing: 100 iterations
- function minimization: 50 iterations
- k-module problem: 50 iterations
- signal processing: 50 iterations

Example:

```bash
conda run -n openevolve_test python -m skydiscover.cli \
  benchmarks/openevolve_parity/function_minimization/initial_program.py \
  benchmarks/openevolve_parity/function_minimization/evaluator.py \
  --config benchmarks/openevolve_parity/function_minimization/config_nvidia.yaml \
  --search adaevolve \
  --output outputs/nvidia_full/function_minimization_adaevolve
```

Switch `--search adaevolve` to `--search evox` for EvoX.

## Export Convergence

After a run, export a FullReplacementMCTS-style per-iteration trace:

```bash
conda run -n openevolve_test python scripts/export_convergence.py \
  outputs/nvidia_full/function_minimization_adaevolve
```

This writes `outputs/.../convergence.jsonl` with `iteration`, `score`, `best_score`, `metrics`, timing fields when available, and program IDs.

The NVIDIA configs set `checkpoint_interval: 1` so EvoX runs persist enough history for this exporter. AdaEvolve also uses `adaevolve_iteration_stats_*.jsonl` when available.

## Files Added

- `benchmarks/math/circle_packing/config_nvidia.yaml`
- `benchmarks/math/signal_processing/config_nvidia.yaml`
- `benchmarks/openevolve_parity/function_minimization/`
- `benchmarks/openevolve_parity/k_module_problem/`
- `scripts/export_convergence.py`
