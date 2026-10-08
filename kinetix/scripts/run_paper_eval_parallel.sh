#!/bin/bash
set -euo pipefail

RUN_PATH=${1:?Usage: $0 <run_path> [output_dir] [num_gpus] [num_evals] [step]}
OUTPUT_DIR=${2:-"eval_outputs_paper"}
NUM_GPUS=${3:-${SLURM_GPUS_ON_NODE:-4}}
NUM_EVALS=${4:-1024}
STEP=${5:--1}
METHODS=${METHODS:-"naive,realtime,vlash,vlash_w_noise,oracle"}

mkdir -p "$OUTPUT_DIR"

echo "Paper eval: $NUM_GPUS GPUs, step=$STEP, num_evals=$NUM_EVALS"
echo "Run path: $RUN_PATH"
echo "Methods: $METHODS"
echo "Output: $OUTPUT_DIR"

for i in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i PYTHONUNBUFFERED=1 uv run python src/eval_flow.py \
        --run-path "$RUN_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --methods-csv "$METHODS" \
        --num-evals "$NUM_EVALS" \
        --step "$STEP" \
        --parallel-index "$i" \
        --parallel-total "$NUM_GPUS" \
        2>&1 | tee "$OUTPUT_DIR/gpu_$i.log" &
done

wait
echo "All jobs completed. Merging results..."
uv run python scripts/merge_results.py --input-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/results.csv"

echo "Plotting Figure 6..."
uv run python scripts/plot_paper_figure.py \
    --input-file "$OUTPUT_DIR/results.csv" \
    --output-file "$OUTPUT_DIR/paper_figure6.png"

echo "Done. Results in $OUTPUT_DIR"
