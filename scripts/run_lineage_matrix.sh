#!/usr/bin/env bash
# Launch AdaEvolve / LineageEvolve runs on cp26 / cp32 / cprect in parallel.
#
# Usage: scripts/run_lineage_matrix.sh [OUT_ROOT] [ITERATIONS] [RUNS] [BENCHES] [METHODS]
#   TAG=viettel|openrouter   which benchmarks/math/<bench>/config_<TAG>_<method>.yaml to use
#   PYTHON=...               interpreter (default: python)
#   FIRST_RUN=1              index of the first run (seeds are 41 + run index)
# The API key variable referenced by the config (VIETTEL_API_KEY / OPENROUTER_API_KEY)
# must be exported. Finished runs (run_summary.json present) are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-outputs/lineage_vs_ada}"; ITERS="${2:-100}"; RUNS="${3:-2}"
BENCHES="${4:-circle_packing circle_packing_n32 circle_packing_rect}"
METHODS="${5:-adaevolve lineageevolve}"
TAG="${TAG:-viettel}"; PY="${PYTHON:-python}"; FIRST="${FIRST_RUN:-1}"
for b in $BENCHES; do for m in $METHODS; do for r in $(seq "$FIRST" $((FIRST + RUNS - 1))); do
  d="$OUT/$b/$m/run_0$r"
  [ -f "$d/run_summary.json" ] && { echo "skip $d"; continue; }
  mkdir -p "$d"
  SKYDISCOVER_SEED=$((41 + r)) nohup "$PY" -m skydiscover.cli \
    "benchmarks/math/$b/initial_program.py" "benchmarks/math/$b/evaluator.py" \
    --config "benchmarks/math/$b/config_${TAG}_$m.yaml" \
    --iterations "$ITERS" --output "$d" > "$d/stdout.log" 2>&1 &
  echo "started $d (pid $!)"
done; done; done
wait
