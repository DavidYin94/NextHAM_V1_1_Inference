import torch
import numpy as np
import sys
import argparse

def compare_tensors(t1, t2, name):
    if type(t1) != type(t2):
        print(f"[{name}] Type mismatch: {type(t1)} vs {type(t2)}")
        return
    
    if isinstance(t1, torch.Tensor):
        # Convert complex to float for easy abs calculation
        if t1.is_complex():
            diff = (t1 - t2).abs()
        else:
            diff = torch.abs(t1 - t2)
            
        max_diff = diff.max().item()
        mean_diff = diff.float().mean().item()
        print(f"[{name}] Tensor shape: {t1.shape}, Max Diff: {max_diff:.8e}, Mean Diff: {mean_diff:.8e}")
    elif isinstance(t1, np.ndarray):
        diff = np.abs(t1 - t2)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"[{name}] Numpy shape: {t1.shape}, Max Diff: {max_diff:.8e}, Mean Diff: {mean_diff:.8e}")
    elif isinstance(t1, (list, tuple)):
        if len(t1) != len(t2):
            print(f"[{name}] Length mismatch: {len(t1)} vs {len(t2)}")
            return
        
        # Check if list of strings/ints vs list of tensors
        if len(t1) > 0 and isinstance(t1[0], (str, int, float, tuple)):
            if t1 == t2:
                print(f"[{name}] Lists match exactly.")
            else:
                print(f"[{name}] Lists differ! First diff at index {next(i for i, (x, y) in enumerate(zip(t1, t2)) if x != y)}")
        else:
            print(f"[{name}] Comparing list elements...")
            for i in range(len(t1)):
                compare_tensors(t1[i], t2[i], f"{name}[{i}]")
    else:
        if t1 == t2:
            print(f"[{name}] Values match exactly ({type(t1)}).")
        else:
            print(f"[{name}] Values differ: {t1} vs {t2}")

def dump_and_compare(txt_dir, bin_dir):
    print("==================================================")
    print(f"Comparing TXT vs BIN")
    print(f"TXT Dir: {txt_dir}")
    print(f"BIN Dir: {bin_dir}")
    print("==================================================")
    
    # 1. Compare input_inference.pth
    txt_input = f"{txt_dir}/input_inference.pth"
    bin_input = f"{bin_dir}/input_inference.pth"
    
    print("\n--- PREPROCESSED DATA (input_inference.pth) ---")
    try:
        data_txt, label_txt = torch.load(txt_input, weights_only=True, map_location='cpu')
        data_bin, label_bin = torch.load(bin_input, weights_only=True, map_location='cpu')
        
        # data is usually a list: [descriptor_tensor, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, ele_list, output_path]
        print("Data elements:")
        for i, name in enumerate(["descriptor_tensor (Weak H)", "overlap_tensor (SR)", "mask_tensor", "edge_vec", "edge_src", "edge_dst", "ele_list", "output_path"]):
            if i < len(data_txt) and i < len(data_bin):
                if data_txt[i] is None and data_bin[i] is None:
                    print(f"[{name}] Both are None")
                else:
                    compare_tensors(data_txt[i], data_bin[i], name)
    except Exception as e:
        print(f"Error loading input_inference.pth: {e}")

    # 2. Compare output_inference.pth
    txt_output = f"{txt_dir}/output_inference.pth"
    bin_output = f"{bin_dir}/output_inference.pth"
    
    print("\n--- INFERENCE RESULT (output_inference.pth) ---")
    try:
        # output_inference.pth is usually a tuple: (precise_H, precise_H_pred, weak_H, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, ele_list)
        out_txt = torch.load(txt_output, weights_only=True, map_location='cpu')
        out_bin = torch.load(bin_output, weights_only=True, map_location='cpu')
        
        print("Output elements:")
        for i, name in enumerate(["precise_H", "precise_H_pred (Predicted H)", "weak_H", "overlap_tensor", "mask_tensor", "edge_vec", "edge_src", "edge_dst", "ele_list"]):
            if i < len(out_txt) and i < len(out_bin):
                if out_txt[i] is None and out_bin[i] is None:
                    print(f"[{name}] Both are None")
                else:
                    compare_tensors(out_txt[i], out_bin[i], name)
    except Exception as e:
        print(f"Error loading output_inference.pth: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare txt vs bin inference data.")
    parser.add_argument("txt_dir", type=str, help="Directory containing txt results")
    parser.add_argument("bin_dir", type=str, help="Directory containing bin results")
    
    args = parser.parse_args()
    dump_and_compare(args.txt_dir, args.bin_dir)
