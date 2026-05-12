#!/bin/bash

# Define base directory and target directories
BASE_DIR=$(pwd)
TARGET_DIR="${BASE_DIR}/get_hs_res/si"
DATA_DIR="${BASE_DIR}/data"
DATASET_DIR="${BASE_DIR}/datasets"

# Create necessary directories
mkdir -p "${DATA_DIR}"
mkdir -p "${DATASET_DIR}"

OUTPUT_PTH="${DATA_DIR}/input_inference_si.pth"
FERMI_ENERGY="6.58"

echo "========================================="
echo " Step 1: Pre-processing (C++ Engine)"
echo "========================================="
cd "${BASE_DIR}/pre_post_process/"
python pre_process.py \
    --read-path "${BASE_DIR}/get_hs_res/si/" \
    --save-path "${BASE_DIR}/data/" \
    --dataset-path "${BASE_DIR}/datasets/"

echo "========================================="
echo " Step 2: Combine Data and Run Inference"
echo "========================================="
# Combine data and run the inference script
python combine_data_infer.py
sh scripts/infer/infer.sh

echo "========================================="
echo " Step 3: Post-processing"
echo "========================================="
# Finalize results and generate plots
python pre_post_process/post_process.py \
    --prediction-path "${BASE_DIR}/data/input_inference_si_out.pth" \
    --stru-file "${TARGET_DIR}/STRU" \
    --data-dir "${TARGET_DIR}/OUT.ABACUS" \
    --save-path "res_si_split/plots/" \
    --fermi ${FERMI_ENERGY}

echo "Pipeline finished successfully!"