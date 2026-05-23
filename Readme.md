# NextHAM: Inference Pipeline

This repository is a lightweight inference branch for the paper **[[ICLR 2026] NextHAM: Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials]**.

Unlike the [main branch](https://github.com/DavidYin94/NextHAM) (which includes full training, testing, and ground-truth accuracy comparisons), this branch focuses strictly on the **inference pipeline**. It is heavily simplified, runs much faster, and is specifically designed for predicting the Hamiltonian of new material structures.

---

## ⚙️ Environment & Compilation Requirements

### Python Environment
The Python dependencies are identical to the main repository. Please refer to the [NextHAM Main Branch](https://github.com/DavidYin94/NextHAM) for detailed Conda/Pip installation instructions.


## 🚀 Quick Start & Usage

### 0. Pre-requisite: Generate Zeroth-Step Hamiltonian
Before running the pipeline, you need to generate the zeroth-step Hamiltonian using the `get_hs` code.
- **Source Code**: [abacus-develop/largescale](https://github.com/goodchong/abacus-develop/tree/largescale)
- Compile and run this code on your target material samples. In our examples, we use a Silicon (Si) system as a sample. You can download the example folder from [here](https://hefei03.nscc-hf.cn:65015/efile/s/w/bmV4dGhhbQ==_7a09882dcb0fe754&) (Extraction code: `bFeV`) and place it in the `get_hs_res/si/` directory.

---

### 🏃‍♂️ Running the Pipeline

**Script**: `full_inference_pipeline_python.sh`

~~~bash
sh full_inference_pipeline_python.sh
~~~

**What this script does:**
1. Runs `pre_process.py` to parse ABACUS outputs and generate the `.pth` graph entirely in Python.
2. Combines the data and runs inference (`infer.sh`).
3. Post-processes the predicted tensors and plots the band structure using `post_process.py`.

---

### 🔍 Customizing the Scripts for Your Materials

If you are evaluating your own structures, open either `.sh` script and modify the variables at the top:

~~~bash
# Modify these paths to point to your specific structure directories
TARGET_DIR="${BASE_DIR}/get_hs_res/YOUR_MATERIAL_DIR"

# Modify the Fermi energy for your specific system (crucial for accurate band plotting)
FERMI_ENERGY="6.58"
~~~

The output plots and final matrices will be saved in the `res_si_split/plots/` (or your defined `--save-path`) directory.

---
*For full training pipelines, evaluation, and accuracy comparisons against ground truth, please visit the [NextHAM Main Repository](https://github.com/DavidYin94/NextHAM).*
