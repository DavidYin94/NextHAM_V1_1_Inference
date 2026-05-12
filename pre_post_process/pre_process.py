import os
import argparse
from pathlib import Path
from ase.io import read, write
from scripts.read_hs_data import HamiltonianDataReader 
from timing_utils import time_execution

@time_execution
def main():
    parser = argparse.ArgumentParser(description="Step 1: Pre-process ABACUS data for NextHAM")
    parser.add_argument("--read-path", type=str, required=True, help="Path to directory containing STRU and OUT.ABACUS")
    parser.add_argument("--save-path", type=str, required=True, help="Directory to save the preprocessed .pth file")
    parser.add_argument("--dataset-path", type=str, required=True, help="Root of .pth file")
    parser.add_argument("--out-dir-name", type=str, default="OUT.ABACUS", help="Name of ABACUS output directory")
    parser.add_argument("--nspin", type=int, default=4, help="Number of spins (1 or 4)")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    out_dir = os.path.join(args.read_path, args.out_dir_name)

    # 1. Handle Structure
    src_stru = os.path.join(out_dir, 'STRU.cif')
    if os.path.exists(src_stru):
        atoms = read(src_stru, format='cif')
    else:
        src_stru = os.path.join(args.read_path, 'STRU')
        atoms = read(src_stru, format='abacus')

    dst_stru = os.path.join(args.save_path, "STRU")
    write(dst_stru, atoms, format='abacus', scaled=False)

    # 2. Extract Matrices
    print(f"Reading data from {out_dir}...")
    read_hs = HamiltonianDataReader(usage="inference", nspin=args.nspin, out_path=args.save_path, 
                                    stru_file=dst_stru, data_dir=out_dir)
    read_hs.read_stru()
    sample_data = read_hs.read_data() # This saves input_inference.pth in save_path
    
    print(f"Pre-processing finished. File saved to: {os.path.join(args.save_path, 'input_inference.pth')}")

    infer_root = os.path.join(args.dataset_path, 'infer_ori.txt')

    file_infer_root = open(infer_root, 'w')
    file_infer_root.write(os.path.join(args.save_path, 'input_inference.pth')+'\n')
    file_infer_root.close()

    print("Root saved to: "+infer_root)


if __name__ == "__main__":
    main()
