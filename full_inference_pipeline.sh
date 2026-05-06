#!/bin/bash

# Get the absolute path of the current script
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Current working directory: ${BASE_DIR}"

# 1. Pre-process
cd "${BASE_DIR}/pre_post_process/"
python pre_process.py \
    --read-path "${BASE_DIR}/get_hs_res/8-atom/" \
    --save-path "${BASE_DIR}/data/" \
    --dataset-path "${BASE_DIR}/datasets/"

# 2. Combine data and run inference
cd "${BASE_DIR}"
python combine_data_infer.py
sh scripts/infer/infer.sh

# 3. Post-process
cd "${BASE_DIR}/pre_post_process/"
python post_process.py \
    --prediction-path "${BASE_DIR}/data/input_inference_out.pth" \
    --stru-file "${BASE_DIR}/get_hs_res/8-atom/STRU" \
    --data-dir "${BASE_DIR}/get_hs_res/8-atom/OUT.ABACUS" \
    --save-path "res_8_split/plots/" \
    --fermi 2.1700249049

echo "All steps completed successfully!"