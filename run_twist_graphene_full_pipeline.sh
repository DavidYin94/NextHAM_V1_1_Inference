#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/zjdai/software/large_scale_inference/NextHAM_V1_1_Inference"
INPUT_DIR="$BASE_DIR/get_hs_res/twist-graphene_vasp_nspin4"
WORK_DIR="$BASE_DIR/data_twist_graphene_vasp_nspin4"
DATASET_DIR="$BASE_DIR/datasets"
LOG_DIR="$BASE_DIR/test_res/twist_graphene_vasp_nspin4"
BACKUP_DIR="$BASE_DIR/.pipeline_backup_twist_graphene_vasp_nspin4"

mkdir -p "$WORK_DIR" "$LOG_DIR" "$BACKUP_DIR"

restore_dataset_files() {
    if [ -f "$BACKUP_DIR/infer.txt.bak" ]; then
        mv -f "$BACKUP_DIR/infer.txt.bak" "$DATASET_DIR/infer.txt"
    else
        rm -f "$DATASET_DIR/infer.txt"
    fi

    if [ -f "$BACKUP_DIR/infer_ori.txt.bak" ]; then
        mv -f "$BACKUP_DIR/infer_ori.txt.bak" "$DATASET_DIR/infer_ori.txt"
    else
        rm -f "$DATASET_DIR/infer_ori.txt"
    fi
}

trap restore_dataset_files EXIT

rm -f "$BACKUP_DIR/infer.txt.bak" "$BACKUP_DIR/infer_ori.txt.bak"

if [ -f "$DATASET_DIR/infer.txt" ]; then
    cp "$DATASET_DIR/infer.txt" "$BACKUP_DIR/infer.txt.bak"
fi

if [ -f "$DATASET_DIR/infer_ori.txt" ]; then
    cp "$DATASET_DIR/infer_ori.txt" "$BACKUP_DIR/infer_ori.txt.bak"
fi

echo "[1/3] Pre-processing $INPUT_DIR"
cd "$BASE_DIR"
python pre_post_process/pre_process.py \
    --read-path "$INPUT_DIR" \
    --save-path "$WORK_DIR" \
    --dataset-path "$DATASET_DIR" \
    --nspin 4

echo "[2/3] Combining inference data"
python combine_data_infer.py

echo "[3/3] Running inference"
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer.py \
    --output-dir "$LOG_DIR" \
    --model-name 'graph_attention_transformer_nonlinear_materials_ham_soc' \
    --input-irreps '64x0e' \
    --radius 8.0 \
    --is-accurate-label \
    --trace-out-len 81 \
    --batch-size 1 \
    --eval-batch-size 1 \
    --weight-decay 0 \
    --num-basis 64 \
    --workers 0 \
    --with-trace \
    --energy-weight 1 \
    --force-weight 80 \
    --test-interval 10000 \
    --target 'hamiltonian' \
    --target-blocks-type 'all' \
    --checkpoint-path1 "$BASE_DIR/pretrained_models/model_range0_curr.pth.tar" \
    --checkpoint-path2 "$BASE_DIR/pretrained_models/model_range1_curr.pth.tar" \
    --checkpoint-path3 "$BASE_DIR/pretrained_models/model_range2_curr.pth.tar" \
    --checkpoint-path4 "$BASE_DIR/pretrained_models/model_range3_curr.pth.tar"

echo
echo "Pipeline finished."
echo "Input PTH:  $WORK_DIR/input_inference.pth"
echo "Output PTH: $WORK_DIR/input_inference_out.pth"
echo "Log dir:    $LOG_DIR"
