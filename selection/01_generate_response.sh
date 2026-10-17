# Step 1: Generate rollout responses
# Produces responses.jsonl with n_per_prompt rollouts per prompt.

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

DATASET=dapo-math-14k
BASE_MODEL=Qwen3-4B-Warmup

MODEL_PATH=$ROOT/models/$BASE_MODEL
INPUT_FILE=$ROOT/data/$DATASET/train.parquet
OUTPUT_FILE=$ROOT/artifacts/outputs/$DATASET/$BASE_MODEL/responses.jsonl

mkdir -p $(dirname $OUTPUT_FILE)

python3 $ROOT/selection/generate_response.py \
  --model_path $MODEL_PATH \
  --input_file $INPUT_FILE \
  --output_file $OUTPUT_FILE \
  --n_per_prompt 8 \
  --max_tokens 4096 \
  --tensor_parallel_size 8 \
  --gpu_memory_utilization 0.9 \
  --temperature 1.0 \
  --top_p 1.0 \
  --labeled \
  --stop '[" )\n\n\n\n\n\n\n\n", "</think>"]'
