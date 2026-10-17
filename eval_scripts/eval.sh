#!/bin/bash

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SAVE_ROOT=$ROOT/artifacts

GROUP=0
TASK=math
EXP_NAME=semi_pivottrace
GLOBAL_STEPS=(200 240 280)

TEMPLATE=qwen
DATASET=dapo-math-14k
BASE_MODEL=Qwen3-4B-Warmup
DATA=$ROOT/data/eval/$TASK.parquet

if [[ "$GROUP" == "0" ]]; then
    GPUS=0,1,2,3
else
    GPUS=4,5,6,7
fi

if [[ "$BASE_MODEL" == "Qwen3-4B-Base" || "$BASE_MODEL" == "Qwen3-4B-Warmup" ]]; then
    MAX_TOKENS=4096
else
    MAX_TOKENS=6144
fi

for GLOBAL_STEP in "${GLOBAL_STEPS[@]}"; do
    OUTPUT_DIR=$ROOT/artifacts/results/$DATASET/$BASE_MODEL/$EXP_NAME/$GLOBAL_STEP/$TASK/$GROUP
    MODEL_PATH=$ROOT/artifacts/ckpts/$DATASET/$BASE_MODEL/$EXP_NAME/global_step_$GLOBAL_STEP/actor/huggingface

    mkdir -p $OUTPUT_DIR

    echo "Evaluating GLOBAL_STEP=$GLOBAL_STEP ..."

    CUDA_VISIBLE_DEVICES=$GPUS python $ROOT/eval_scripts/generate_vllm.py \
        --model_path $MODEL_PATH \
        --input_file $DATA \
        --remove_system True \
        --add_oat_evaluate True \
        --max_tokens $MAX_TOKENS \
        --output_file $OUTPUT_DIR/$TASK.jsonl \
        --template $TEMPLATE > $OUTPUT_DIR/$TASK.log
done

