# CORAL Kernel Builder

This benchmark wraps CORAL `examples/kernel_builder`, a CPU simulator task for optimizing a VLIW/SIMD tree-traversal kernel.

Main metric:

- `cycles`: raw simulator cycles, lower is better.
- `combined_score`: `147734 / cycles`, higher is better. This is only a framework-compatible maximization proxy.

No NVIDIA GPU is required for evaluation, but the provided config uses the NVIDIA LLM API.

Run SkyDiscover/AdaEvolve:

```bash
export NVIDIA_API_KEY=...
export CORAL_PYTHON="$(which python)"
python scripts/run_nvidia_parity_matrix.py \
  --runs 5 \
  --benchmarks kbuilder \
  --strategies adaevolve \
  --iterations 300 \
  --output-root outputs/coral_kernel_builder_adaevolve_5runs_300iter
```

For smoke tests, prefix with `CORAL_KERNEL_BUILDER_TUNE=1` and use `--runs 1 --iterations 1 --fail-fast`. Tune mode is not comparable to real-mode cycle counts.
