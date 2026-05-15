#!/bin/bash

# Define base directory and target directories
BASE_DIR=$(pwd)
TARGET_DIR="${BASE_DIR}/get_hs_res/si"
DATA_DIR="${BASE_DIR}/data"
DATASET_DIR="${BASE_DIR}/datasets"

# Create necessary directories
mkdir -p "${DATA_DIR}"
mkdir -p "${DATASET_DIR}"

OUTPUT_PTH="${DATA_DIR}/cpp_input_inference_si.pth"
FERMI_ENERGY="6.58"

echo "========================================="
echo " Step 1: Pre-processing (C++ Engine)"
echo "========================================="
# Run the C++ pre-processing executable
./pre_post_process/cpp/build/nextham_preprocess \
    "${TARGET_DIR}/STRU" \
    "${TARGET_DIR}/OUT.ABACUS/" \
    4 \
    8.0 \
    "${OUTPUT_PTH}"

echo "========================================="
echo " Step 1.5: Generating infer_ori.txt"
echo "========================================="
# Write the generated .pth file path into datasets/infer_ori.txt
INFER_ROOT="${DATASET_DIR}/infer_ori.txt"
echo "${OUTPUT_PTH}" > "${INFER_ROOT}"
echo "Root saved to: ${INFER_ROOT}"

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
    --prediction-path "${BASE_DIR}/data/cpp_input_inference_si_out.pth" \
    --stru-file "${DATA_DIR}/STRU" \
    --data-dir "${TARGET_DIR}/OUT.ABACUS" \
    --save-path "res_si_split/plots/" \
    --fermi ${FERMI_ENERGY}

echo "Pipeline finished successfully!"
