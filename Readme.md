# NextHAM: Inference Pipeline

This repository is a lightweight inference branch for the paper **[[ICLR 2026] NextHAM: Advancing Universal Deep Learning for Electronic-Structure Hamiltonian Prediction of Materials]**.

Unlike the [main branch](https://github.com/DavidYin94/NextHAM) (which includes full training, testing, and ground-truth accuracy comparisons), this branch focuses strictly on the **inference pipeline**. It is heavily simplified, runs much faster, and is specifically designed for predicting the Hamiltonian of new material structures.

---

## ⚙️ Environment & Compilation Requirements

### 1. Python Environment
The Python dependencies are identical to the main repository. Please refer to the [NextHAM Main Branch](https://github.com/DavidYin94/NextHAM) for detailed Conda/Pip installation instructions.

### 2. C++ Preprocessor Compilation (Highly Recommended)
Data parsing and graph generation can be a major bottleneck. To dramatically accelerate this step, we provide a C++ preprocessor. **We highly recommend compiling and using this C++ program for inference**, though a slower, pure-Python alternative is also provided.

**Dependencies:**
* C++17 compatible compiler (e.g., GCC 7+) and CMake (>= 3.16).
* **LibTorch** (PyTorch C++ backend, version matching your python env, e.g., v2.2.0, cxx11 ABI).
* **Eigen3** (C++ template library for linear algebra. Header-only).

**Compilation Guide:**

1. **Download Dependencies**: Download LibTorch and Eigen3 to your local machine and extract them. Note down their absolute paths.
2. **Modify CMakeLists.txt**: Open `pre_post_process/cpp/CMakeLists.txt` and replace the hardcoded paths with your actual absolute paths:
   ~~~cmake
   # Find LibTorch (Update this path!)
   set(CMAKE_PREFIX_PATH "/your/actual/path/to/libtorch")
   find_package(Torch REQUIRED)

   # Find Eigen3 (Update this path!)
   set(EIGEN3_INCLUDE_DIR "/your/actual/path/to/eigen-3.4.0")
   include_directories(${EIGEN3_INCLUDE_DIR})
   ~~~
3. **Build the Executable**:
   ~~~bash
   cd pre_post_process/cpp
   mkdir build && cd build
   cmake ..
   make -j4
   ~~~
   After a successful build, the `nextham_preprocess` executable will be generated inside the `build/` directory. Return to the project root directory before continuing.

---

## 🚀 Quick Start & Usage

### 0. Pre-requisite: Generate Zeroth-Step Hamiltonian
Before running the pipeline, you need to generate the zeroth-step Hamiltonian using the `get_hs` code.
- **Source Code**: [abacus-develop/largescale](https://github.com/goodchong/abacus-develop/tree/largescale)
- Compile and run this code on your target material samples. In our examples, we use an si Si system whose outputs are saved in `get_hs_res/si/`.

---

### 🏃‍♂️ Running the Pipeline

We provide two end-to-end bash scripts to run the pipeline. **We strongly recommend Method 1** as it utilizes the C++ engine for maximum speed. 

#### Method 1: Hybrid Pipeline (Highly Recommended 🌟)
**Script**: `full_inference_pipeline_hybrid.sh`

This method uses the compiled C++ executable for ultra-fast data parsing, followed by Python for PyTorch inference and post-processing.

~~~bash
# Ensure you have compiled the C++ executable first!
sh full_inference_pipeline_hybrid.sh
~~~

**What this script does:**
1. Runs the C++ parser (`nextham_preprocess`) to generate the `.pth` graph and manually routes the output path to `datasets/infer_ori.txt`.
2. Combines the data and runs the Graph Attention Transformer inference (`infer.sh`).
3. Post-processes the predicted tensors back to ABACUS format, supplements weak labels, and plots the band structure using `post_process.py`.

#### Method 2: Pure Python Pipeline (Alternative 🐢)
**Script**: `full_inference_pipeline_python.sh`

If you are unable to compile the C++ engine, you can use the pure Python fallback. **Note: This method is significantly slower.**

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
FERMI_ENERGY="2.1700249049"
~~~

The output plots and final matrices will be saved in the `res_si_split/plots/` (or your defined `--save-path`) directory.

---
*For full training pipelines, evaluation, and accuracy comparisons against ground truth, please visit the [NextHAM Main Repository](https://github.com/DavidYin94/NextHAM).*