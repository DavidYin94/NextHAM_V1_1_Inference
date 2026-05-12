import os
import argparse
import torch
from scripts.output_hs_data import transform_abacus_hamiltion_data
from scripts.add_element import add_hs_matrix
from scripts.plot_band import plot_band
from timing_utils import time_execution

@time_execution
def main():
    parser = argparse.ArgumentParser(description="Step 3: Post-process predictions and plot bands")
    parser.add_argument("--prediction-path", type=str, required=True, help="Path to output_inference.pth from infer.py")
    parser.add_argument("--stru-file", type=str, required=True, help="Path to the STRU file")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to original ABACUS output dir (containing srs1_nao.csr)")
    parser.add_argument("--save-path", type=str, default="res_final_plots", help="Directory to save final results")
    parser.add_argument("--nspin", type=int, default=4, help="Number of spins")
    parser.add_argument("--fermi", type=float, default=2.1700249049, help="Fermi level for plotting")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    print("Step A: Transforming back to ABACUS format...")
    transform_data = transform_abacus_hamiltion_data("inference", args.nspin, args.stru_file, args.prediction_path, args.save_path)
    transform_data.read_stru()
    transform_data.get_hs_data()

    print("\nStep B: Supplementing weak label elements...")
    predict_hr_cut = os.path.join(args.save_path, 'predict_hr_cut')
    add_elem = add_hs_matrix(args.nspin, args.stru_file, predict_hr_cut, args.data_dir, args.save_path)
    add_elem.read_stru()
    add_elem.add_matrxi_element()

    print("\nStep C: Calculating and Plotting Band Structure...")
    hr_tot = os.path.join(args.save_path, 'predict_hr_tot')
    src_overlap = os.path.join(args.data_dir, 'srs1_nao.csr')
    
    try:
        band_plot = plot_band(args.nspin, args.stru_file, hr_tot, src_overlap, -10, 10, args.fermi, args.save_path)
        band_plot.cal_band()
        band_plot.plot_pic()
        print(f"\nSUCCESS! Post-processing complete. Results in: {args.save_path}")
    except Exception as e:
        print(f"\nError during band plotting: {e}")

if __name__ == "__main__":
    main()
