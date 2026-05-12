import numpy as np
import struct
import glob
import os
import argparse
from scipy.sparse import csr_matrix

def parse_dat_file(filepath, is_binary, nspin):
    """Parses a single rank's block .dat file (text or binary)."""
    data = {}
    
    if not is_binary:
        with open(filepath, 'r') as f:
            step_line = f.readline()
            if not step_line: return None, 0
            step = int(step_line.split()[-1])
            n_ap_line = f.readline()
            n_ap = int(n_ap_line.split()[-1])
            
            for _ in range(n_ap):
                pair_line = f.readline()
                if not pair_line: break
                parts = pair_line.split()
                ai, aj, rs, cs, nr = map(int, parts[1:])
                pair_key = (ai, aj)
                if pair_key not in data:
                    data[pair_key] = []
                
                # Read RowIdx and ColIdx
                row_idx_line = f.readline()
                col_idx_line = f.readline()
                row_idx = [int(x) for x in row_idx_line.split()[1:]]
                col_idx = [int(x) for x in col_idx_line.split()[1:]]

                for _ in range(nr):
                    r_line = f.readline()
                    r_parts = r_line.split()
                    rx, ry, rz = map(int, r_parts[1:])
                    
                    block_size = rs * cs
                    block_data = []
                    while len(block_data) < block_size:
                        line = f.readline()
                        if not line: break
                        parts = line.split()
                        if not parts: continue
                        if parts[0] in ['R:', 'Pair:']:
                            break
                        
                        if nspin == 4:
                            for i in range(0, len(parts), 2):
                                block_data.append(complex(float(parts[i]), float(parts[i+1])))
                        else:
                            for p in parts:
                                block_data.append(float(p))
                    
                    mat = np.array(block_data).reshape((rs, cs))
                    data[pair_key].append(((rx, ry, rz), row_idx, col_idx, mat))
    else:
        with open(filepath, 'rb') as f:
            raw_header = f.read(8)
            if not raw_header: return None, 0
            step, n_ap = struct.unpack('ii', raw_header)
            
            for _ in range(n_ap):
                raw_pair = f.read(20)
                ai, aj, rs, cs, nr = struct.unpack('iiiii', raw_pair)
                pair_key = (ai, aj)
                if pair_key not in data:
                    data[pair_key] = []
                    
                row_idx = np.frombuffer(f.read(rs * 4), dtype=np.int32).tolist()
                col_idx = np.frombuffer(f.read(cs * 4), dtype=np.int32).tolist()

                for _ in range(nr):
                    raw_R = f.read(12)
                    rx, ry, rz = struct.unpack('iii', raw_R)
                    
                    block_size = rs * cs
                    if nspin == 4:
                        raw_data = f.read(block_size * 16)
                        block_data = np.frombuffer(raw_data, dtype=np.complex128)
                    else:
                        raw_data = f.read(block_size * 8)
                        block_data = np.frombuffer(raw_data, dtype=np.float64)
                        
                    mat = block_data.reshape((rs, cs))
                    data[pair_key].append(((rx, ry, rz), row_idx, col_idx, mat))
                    
    return data, step

def convert(out_dir, prefix, output_filename, nspin, is_binary, threshold=1e-10, ref_csr=None):
    files = glob.glob(os.path.join(out_dir, f"{prefix}_*.dat"))
    if not files:
        print(f"No files found with prefix {prefix}")
        return

    all_data = {}
    step = 0
    
    # Track max dim dynamically
    max_idx = -1

    print(f"Reading {len(files)} files...")
    for f in files:
        process_data, s = parse_dat_file(f, is_binary, nspin)
        if process_data is None: continue
        step = s
        for pair, contents in process_data.items():
            for r_vec, row_idx, col_idx, mat in contents:
                if r_vec not in all_data:
                    all_data[r_vec] = []
                all_data[r_vec].append((row_idx, col_idx, mat))
                if row_idx: max_idx = max(max_idx, max(row_idx))
                if col_idx: max_idx = max(max_idx, max(col_idx))

    dim = max_idx + 1
    r_vectors = set(all_data.keys())
    
    if ref_csr:
        with open(ref_csr, 'r') as f:
            f.readline()
            f.readline()
            nr = int(f.readline().split()[-1])
            ref_r = set()
            for _ in range(nr):
                header = f.readline().split()
                if not header: break
                if len(header) < 4: continue
                rx, ry, rz, nnz = map(int, header)
                ref_r.add((rx, ry, rz))
                if nnz == 0:
                    continue
                else:
                    f.readline(); f.readline(); f.readline()
            r_vectors = r_vectors.union(ref_r)

    r_vectors = sorted(list(r_vectors))
    
    print(f"Writing CSR file {output_filename} (Dim: {dim}, R-vectors: {len(r_vectors)})...")
    with open(output_filename, 'w') as f:
        f.write(f"STEP: {step}\n")
        f.write(f"Matrix dimension of H(R): {dim}\n")
        f.write(f"Matrix number of H(R): {len(r_vectors)}\n")
        
        for r in r_vectors:
            matrix_elements = {}
            
            if r in all_data:
                for row_idx, col_idx, mat in all_data[r]:
                    # Apply threshold
                    non_zero = np.abs(mat) > threshold
                    if not np.any(non_zero):
                        continue
                    
                    idx = np.where(non_zero)
                    for i, j in zip(idx[0], idx[1]):
                        row, col = row_idx[i], col_idx[j]
                        key = (row, col)
                        val = mat[i, j]
                        if key in matrix_elements:
                            matrix_elements[key] += val
                        else:
                            matrix_elements[key] = val
            
            if not matrix_elements:
                f.write(f"{r[0]} {r[1]} {r[2]} 0\n")
                continue

            # Need to sort keys primarily by row, then col
            sorted_keys = sorted(matrix_elements.keys(), key=lambda x: (x[0], x[1]))
            rows = [k[0] for k in sorted_keys]
            cols = [k[1] for k in sorted_keys]
            vals = [matrix_elements[k] for k in sorted_keys]

            # Convert to CSR to ensure standard ordering (row-major)
            csr = csr_matrix((vals, (rows, cols)), shape=(dim, dim))
            csr.eliminate_zeros() # Final check
            
            f.write(f"{r[0]} {r[1]} {r[2]} {csr.nnz}\n")
            
            # Values
            if nspin == 4:
                val_str = " ".join([f"({v.real:.8e},{v.imag:.8e})" for v in csr.data])
            else:
                val_str = " ".join([f"{v:.8e}" for v in csr.data])
            f.write(val_str + "\n")
            
            # Indices
            f.write(" ".join(map(str, csr.indices)) + "\n")
            
            # Indptr
            f.write(" ".join(map(str, csr.indptr)) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="OUT.ABACUS")
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--nspin", type=int, default=4)
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--ref", type=str, default=None)
    args = parser.parse_args()
    convert(args.dir, args.prefix, args.out, args.nspin, args.binary, ref_csr=args.ref)
