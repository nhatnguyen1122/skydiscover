# CORAL TriMul Kernel Engineering

This benchmark wraps CORAL `examples/kernel_engineering/trimul` as a CUDA/Triton optimization task. It requires a Linux server with an NVIDIA CUDA GPU, PyTorch CUDA, Triton, and PyYAML.

Install into your experiment environment:

```bash
conda create -n coral_kernel python=3.11 -y
conda activate coral_kernel
pip install -r benchmarks/coral_trimul/requirements.txt
```

Run SkyDiscover/AdaEvolve on the CORAL TriMul task:

```bash
export NVIDIA_API_KEY=...
export CORAL_PYTHON="$(which python)"
python scripts/run_nvidia_parity_matrix.py \
  --runs 5 \
  --benchmarks trimul \
  --strategies adaevolve \
  --iterations 300 \
  --output-root outputs/coral_trimul_adaevolve_5runs_300iter
```

For a cheap server smoke test, prefix the command with `CORAL_TRIMUL_QUICK=1` and use `--runs 1 --iterations 1 --fail-fast`. Do not use quick mode for final comparisons.

On smaller GPUs, use the explicit medium subset for comparable experiments:

```bash
CORAL_TRIMUL_SUBSET=medium python scripts/run_nvidia_parity_matrix.py ...
```

The medium subset keeps all CORAL tests and benchmarks with `seqlen <= 256` and avoids the full suite's 768/1024 sequence cases.
