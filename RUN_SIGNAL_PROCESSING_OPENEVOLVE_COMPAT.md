To run AdaEvolve and EvoX on signal processing that is comparable with FullReplacementMCTS and AdaEvolveMCTS, use these commands:

  Run AdaEvolve 5 times:

  cd "/Users/nhatnguyen/Documents/Training Lab cô Bình/TAPL/code/workspace/skydiscover"

  conda run -n openevolve_test python scripts/run_nvidia_parity_matrix.py \
    --runs 5 \
    --benchmarks sp_oe \
    --strategies adaevolve \
    --output-root outputs/adaevolve_sp_oe_5runs \
    --resume

  Run EvoX 5 times:

  cd "/Users/nhatnguyen/Documents/Training Lab cô Bình/TAPL/code/workspace/skydiscover"

  conda run -n openevolve_test python scripts/run_nvidia_parity_matrix.py \
    --runs 5 \
    --benchmarks sp_oe \
    --strategies evox \
    --output-root outputs/evox_sp_oe_5runs \
    --resume

  Run both in one matrix:

  conda run -n openevolve_test python scripts/run_nvidia_parity_matrix.py \
    --runs 5 \
    --benchmarks sp_oe \
    --strategies adaevolve,evox \
    --output-root outputs/skydiscover_sp_oe_5runs \
    --resume