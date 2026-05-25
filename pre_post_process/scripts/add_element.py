# 把截断半径以外的数据用弱标签哈密顿量数据补全，获取新的数据
import numpy as np
import os 
import re
import struct
import glob
import shutil
from typing import Optional, Union, List, Tuple, Dict, Any
from ase.io import read, write
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
import sys
from pathlib import Path
try:
    from timing_utils import time_execution
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from timing_utils import time_execution


class XR_matrix:
    def __init__(self, nspin, path: str, matrix_type: str = 'hrs'):
        self.nspin = nspin
        self.path = Path(path)
        self.matrix_type = matrix_type
        self.read_file()

    def _parse_dat_file(self, filepath: str, is_binary: bool) -> Tuple[Optional[Dict], int]:
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
                    parts = f.readline().split()
                    ai, aj, rs, cs, nr = map(int, parts[1:])
                    row_idx = [int(x) for x in f.readline().split()[1:]]
                    col_idx = [int(x) for x in f.readline().split()[1:]]
                    for _ in range(nr):
                        r_parts = f.readline().split()
                        rx, ry, rz = map(int, r_parts[1:])
                        block_size = rs * cs
                        block_data = []
                        while len(block_data) < block_size:
                            line = f.readline()
                            if not line: break
                            parts = line.split()
                            if not parts: continue
                            if parts[0] in ['R:', 'Pair:']: break
                            if self.nspin == 4:
                                for i in range(0, len(parts), 2):
                                    block_data.append(complex(float(parts[i]), float(parts[i+1])))
                            else:
                                for p in parts: block_data.append(float(p))
                        mat = np.array(block_data).reshape((rs, cs))
                        data.setdefault((ai, aj), []).append(((rx, ry, rz), row_idx, col_idx, mat))
        else:
            with open(filepath, 'rb') as f:
                raw_header = f.read(8)
                if not raw_header: return None, 0
                step, n_ap = struct.unpack('ii', raw_header)
                for _ in range(n_ap):
                    ai, aj, rs, cs, nr = struct.unpack('iiiii', f.read(20))
                    row_idx = np.frombuffer(f.read(rs * 4), dtype=np.int32).tolist()
                    col_idx = np.frombuffer(f.read(cs * 4), dtype=np.int32).tolist()
                    for _ in range(nr):
                        rx, ry, rz = struct.unpack('iii', f.read(12))
                        block_size = rs * cs
                        dtype = np.complex128 if self.nspin == 4 else np.float64
                        block_data = np.frombuffer(f.read(block_size * np.dtype(dtype).itemsize), dtype=dtype)
                        mat = block_data.reshape((rs, cs))
                        data.setdefault((ai, aj), []).append(((rx, ry, rz), row_idx, col_idx, mat))
        return data, step

    def _read_all_block_matrices(self, dir_path: Path, prefix: str):
        """Reads distributed dense blocks and assembles them into global dense matrices."""
        files = glob.glob(str(dir_path / f"{prefix}_*.dat"))
        if not files: return None
        is_binary = False
        with open(files[0], 'rb') as f:
            if f.read(4) != b'STEP': is_binary = True
        all_data = {}; max_idx = -1
        for f in files:
            process_data, step = self._parse_dat_file(f, is_binary)
            if process_data is None: continue
            for contents in process_data.values():
                for r_vec, row_idx, col_idx, mat in contents:
                    all_data.setdefault(r_vec, []).append((row_idx, col_idx, mat))
                    if row_idx: max_idx = max(max_idx, max(row_idx))
                    if col_idx: max_idx = max(max_idx, max(col_idx))
        dim = max_idx + 1; self.basis_num = dim
        r_vectors = sorted(all_data.keys()); self.R_num = len(r_vectors)
        self.R_direct_coor = np.array(r_vectors)
        dtype = complex if self.nspin == 4 else float
        self.XR = np.zeros([self.R_num, dim, dim], dtype=dtype)
        for iR, r in enumerate(r_vectors):
            for row_idx, col_idx, mat in all_data[r]:
                for i in range(len(row_idx)):
                    for j in range(len(col_idx)):
                        self.XR[iR, row_idx[i], col_idx[j]] += mat[i, j]
        return True

    @time_execution
    def read_file(self):
        if self.path.is_file():
            csr_file = self.path
        elif self.path.is_dir():
            csr_file = self.path / f"{self.matrix_type}1_nao.csr"
            if not csr_file.exists():
                prefix = 'hrs_block_up' if self.matrix_type == 'hrs' else 'srs_block'
                if self._read_all_block_matrices(self.path, prefix):
                    return
                raise FileNotFoundError(f"Neither {csr_file.name} nor {prefix}_*.dat found in {self.path}")
        else:
            raise FileNotFoundError(f"Path does not exist: {self.path}")

        # Auto-detect binary or text mode
        is_binary = False
        with open(csr_file, 'rb') as f:
            header = f.read(4)
            if header != b'STEP':
                is_binary = True
        
        if not is_binary:
            with open(csr_file, 'r') as fread:
                fread.readline() # STEP
                line = fread.readline()
                self.basis_num = int(line.split()[-1])
                line = fread.readline()
                self.R_num = int(line.split()[-1])
                self.R_direct_coor = np.zeros([self.R_num, 3], dtype=int)
                if self.nspin != 4:
                    self.XR = np.zeros([self.R_num, self.basis_num, self.basis_num], dtype=float)
                else:
                    self.XR = np.zeros([self.R_num, self.basis_num, self.basis_num], dtype=complex)

                for iR in range(self.R_num):
                    line = fread.readline().split()
                    self.R_direct_coor[iR, 0] = int(line[0])
                    self.R_direct_coor[iR, 1] = int(line[1])
                    self.R_direct_coor[iR, 2] = int(line[2])
                    data_size = int(line[3])
                    
                    if self.nspin != 4:
                        data = np.zeros((data_size,), dtype=float)
                    else:
                        data = np.zeros((data_size,), dtype=complex)

                    indices = np.zeros((data_size,), dtype=int)
                    indptr = np.zeros((self.basis_num+1,), dtype=int)

                    if data_size != 0:
                        if self.nspin != 4:
                            line = fread.readline().split()
                            if (len(line) != data_size):
                                print("size = ", len(line), " data_size = ", data_size)
                            for index in range(data_size):
                                data[index] = float(line[index])
                        else:
                            line = re.findall('[(](.*?)[])]', fread.readline())
                            for index in range(data_size):
                                value = line[index].split(',')
                                data[index] = complex( float(value[0]), float(value[1]) ) 

                        line = fread.readline().split()
                        for index in range(data_size):
                            indices[index] = int(line[index])

                        line = fread.readline().split()
                        for index in range(self.basis_num+1):
                            indptr[index] = int(line[index])

                    self.XR[iR] = csr_matrix((data, indices, indptr), shape=(self.basis_num, self.basis_num)).toarray()
        else:
            with open(csr_file, 'rb') as f:
                # Binary format: step(int), nlocal(int), nR(int)
                raw_header = f.read(12)
                if not raw_header: return
                step, nlocal, nR = struct.unpack('iii', raw_header)
                self.basis_num = nlocal
                self.R_num = nR
                self.R_direct_coor = np.zeros([self.R_num, 3], dtype=int)
                if self.nspin != 4:
                    self.XR = np.zeros([self.R_num, self.basis_num, self.basis_num], dtype=float)
                else:
                    self.XR = np.zeros([self.R_num, self.basis_num, self.basis_num], dtype=complex)
                
                for iR in range(nR):
                    raw_R = f.read(16) # rx, ry, rz, nnz
                    if not raw_R: break
                    rx, ry, rz, nnz = struct.unpack('iiii', raw_R)
                    self.R_direct_coor[iR] = [rx, ry, rz]
                    
                    if nnz == 0:
                        f.read((self.basis_num + 1) * 4)
                        continue
                    
                    if self.nspin == 4:
                        data = np.frombuffer(f.read(nnz * 16), dtype=complex)
                    else:
                        data = np.frombuffer(f.read(nnz * 8), dtype=float)
                    
                    indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)
                    indptr = np.frombuffer(f.read((self.basis_num + 1) * 4), dtype=np.int32)
                    
                    self.XR[iR] = csr_matrix((data, indices, indptr), shape=(self.basis_num, self.basis_num)).toarray()


class add_hs_matrix:

    def __init__(self, nspin, stru_file, hr1, hr2, save_path):
        self.nspin = nspin
        self.stru_file = stru_file
        self.hr1 = hr1
        self.hr2 = hr2
        self.save_path = save_path
        self.max_dense_gib = float(os.environ.get("NEXTHAM_MAX_DENSE_GIB", "64"))
        self.orb_origin = {'H': 5,   'He': 5,  'Li': 7,  'Be': 7,  'B': 13,  'C': 13,  'N': 13,  'O': 13,  'F': 13, 'Ne': 13, 
                           'Na': 15, 'Mg': 15, 'Al': 13, 'Si': 13, 'P': 13,  'S': 13,  'Cl': 13, 'Ar': 13, 'K': 15, 
                           'Sc': 27, 'V': 27,  'Fe': 27, 'Co': 27, 'Ni': 27, 'Cu': 27, 'Zn': 27, 'Ga': 25, 'Ge': 25, 
                           'Br': 13, 'Y': 27,  'Nb': 27, 'Mo': 27, 'Pd': 25, 'Ag': 27, 'Cd': 27, 'In': 25, 'Sn': 25, 
                           'Sb': 25, 'Te': 25, 'I': 13,  'Xe': 13, 'Hf': 27, 'Ta': 27, 'Re': 27, 'Pt': 27, 'Au': 27, 
                           'Hg': 27, 'Tl': 25, 'Pb': 25, 'Bi': 25, 'Ca': 15, 'Ti': 27, 'Cr': 27, 'Mn': 27, 'Kr': 13, 
                           'Rb': 15, 'Sr': 15, 'Zr': 27, 'Tc': 27, 'Ru': 27, 'Rh': 27, 'Cs': 15, 'Ba': 15, 'W': 27, 
                           'Os': 27, 'Ir': 27, 'As': 13, 'Se': 13}        

    @time_execution
    def read_stru(self):
        self.atoms = read(self.stru_file, format='abacus')
        self.atoms.wrap()
        # 创建一个字典，用以存储结构的原子指标和对应的哈密顿量起始和终止指标，字典类型为 ii item {element: [start, end]}
        index_dict = {}
        current_index = 0
        for ii, atom in enumerate(self.atoms):
            element = atom.symbol
            n_orbitals = self.orb_origin[element]
            start = current_index
            if self.nspin == 4:
                n_orbitals = n_orbitals * 2
            end = current_index + n_orbitals
            index_dict[ii] = [start, end]
            current_index = end  # 更新到下一个原子的起始轨道索引
        self.basis_num = current_index
        # print('index_dict:', index_dict)
        return index_dict

    def _estimated_dense_gib(self, r_num: int) -> float:
        bytes_per_value = 16 if self.nspin == 4 else 8
        total_bytes = r_num * self.basis_num * self.basis_num * bytes_per_value
        return total_bytes / (1024 ** 3)

    def _copy_predicted_cut_to_total(self):
        hr_add_file = os.path.join(self.save_path, 'predict_hr_tot')
        shutil.copyfile(self.hr1, hr_add_file)
        print(
            "Step B fallback: system is too large for dense weak-label supplementation. "
            "Using predict_hr_cut directly as predict_hr_tot.",
            flush=True,
        )
        print(f"Saved fallback Hamiltonian to: {hr_add_file}", flush=True)

    def _detect_binary(self, path: Union[str, Path]) -> bool:
        with open(path, 'rb') as f:
            return f.read(4) != b'STEP'

    def _ensure_weak_hr_csr(self) -> str:
        """Return a weak-label HR CSR path, converting ABACUS block files if needed."""
        hr2_path = Path(self.hr2)
        if hr2_path.is_file():
            return str(hr2_path)
        if not hr2_path.is_dir():
            raise FileNotFoundError(f"Weak-label HR path does not exist: {self.hr2}")

        csr_file = hr2_path / 'hrs1_nao.csr'
        if csr_file.exists():
            return str(csr_file)

        block_files = sorted(hr2_path.glob('hrs_block_up_*.dat'))
        if not block_files:
            raise FileNotFoundError(f"Neither hrs1_nao.csr nor hrs_block_up_*.dat found in {hr2_path}")

        print(f"Converting weak-label ABACUS block files to sparse CSR: {csr_file}", flush=True)
        try:
            from convert_block_to_csr import convert
        except ImportError:
            sys.path.append(str(Path(__file__).parent))
            from convert_block_to_csr import convert
        convert(
            str(hr2_path),
            'hrs_block_up',
            str(csr_file),
            self.nspin,
            self._detect_binary(block_files[0]),
            ref_csr=self.hr1,
        )
        return str(csr_file)

    def _read_csr_text_header(self, fh) -> Tuple[int, int]:
        fh.readline()  # STEP
        dim = int(fh.readline().split()[-1])
        r_num = int(fh.readline().split()[-1])
        return dim, r_num

    def _parse_csr_data_line(self, line: str, nnz: int) -> np.ndarray:
        if nnz == 0:
            dtype = np.complex128 if self.nspin == 4 else np.float64
            return np.empty((0,), dtype=dtype)
        if self.nspin == 4:
            values = (
                complex(float(m.group(1)), float(m.group(2)))
                for m in re.finditer(r'\(([^,]+),([^)]+)\)', line)
            )
            data = np.fromiter(values, dtype=np.complex128, count=nnz)
        else:
            data = np.fromstring(line, sep=' ', dtype=np.float64, count=nnz)
        if data.size != nnz:
            raise ValueError(f"CSR data length mismatch: expected {nnz}, got {data.size}")
        return data

    def _iter_csr_text(self, path: Union[str, Path]):
        """Yield one sparse H(R) matrix at a time from the text CSR format."""
        with open(path, 'r') as fh:
            dim, r_num = self._read_csr_text_header(fh)
            dtype = np.complex128 if self.nspin == 4 else np.float64
            for _ in range(r_num):
                header = fh.readline().split()
                if not header:
                    break
                rx, ry, rz = map(int, header[:3])
                nnz = int(header[3])
                r_vec = (rx, ry, rz)
                if nnz == 0:
                    yield r_vec, csr_matrix((dim, dim), dtype=dtype), dim
                    continue

                data = self._parse_csr_data_line(fh.readline(), nnz)
                indices = np.fromstring(fh.readline(), sep=' ', dtype=np.int64, count=nnz)
                indptr = np.fromstring(fh.readline(), sep=' ', dtype=np.int64, count=dim + 1)
                if indices.size != nnz:
                    raise ValueError(f"CSR indices length mismatch for R={r_vec}: expected {nnz}, got {indices.size}")
                if indptr.size != dim + 1:
                    raise ValueError(f"CSR indptr length mismatch for R={r_vec}: expected {dim + 1}, got {indptr.size}")
                matrix = csr_matrix((data, indices, indptr), shape=(dim, dim))
                yield r_vec, matrix, dim

    def _write_values_line(self, fh, data: np.ndarray, chunk_size: int = 100000):
        first = True
        for start in range(0, data.size, chunk_size):
            chunk = data[start:start + chunk_size]
            if self.nspin == 4:
                text = " ".join("({:.8e},{:.8e})".format(v.real, v.imag) for v in chunk)
            else:
                text = " ".join("{:.8e}".format(v) for v in chunk)
            if text:
                fh.write(text if first else " " + text)
                first = False
        fh.write("\n")

    def _write_int_line(self, fh, values: np.ndarray, chunk_size: int = 200000):
        first = True
        for start in range(0, values.size, chunk_size):
            text = " ".join(map(str, values[start:start + chunk_size]))
            if text:
                fh.write(text if first else " " + text)
                first = False
        fh.write("\n")

    def _write_csr_text_matrix(self, fh, r_vec: Tuple[int, int, int], matrix: csr_matrix):
        matrix = matrix.tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        fh.write(f"{r_vec[0]} {r_vec[1]} {r_vec[2]} {matrix.nnz}\n")
        if matrix.nnz == 0:
            return
        self._write_values_line(fh, matrix.data)
        self._write_int_line(fh, matrix.indices)
        self._write_int_line(fh, matrix.indptr)

    def _query_near_atom_pairs(self, r_vec: Tuple[int, int, int], r_cut: float):
        positions = self.atoms.positions
        shift = r_vec[0] * self.atoms.cell[0] + r_vec[1] * self.atoms.cell[1] + r_vec[2] * self.atoms.cell[2]
        tree_i = cKDTree(positions)
        tree_j = cKDTree(positions + shift)
        return tree_i.query_ball_tree(tree_j, r_cut)

    def _block_membership_mask(self, matrix: csr_matrix, index_dict: Dict[int, List[int]], pairs_by_atom) -> np.ndarray:
        """Mask CSR entries that belong to atom-pair blocks within R_cut."""
        matrix = matrix.tocsr()
        mask = np.zeros(matrix.nnz, dtype=bool)
        for atom_i, atom_js in enumerate(pairs_by_atom):
            if not atom_js:
                continue
            row_start, row_end = index_dict[atom_i]
            col_ranges = np.array([index_dict[atom_j] for atom_j in atom_js], dtype=np.int64)
            order = np.argsort(col_ranges[:, 0])
            col_starts = col_ranges[order, 0]
            col_ends = col_ranges[order, 1]
            for row in range(row_start, row_end):
                start = matrix.indptr[row]
                end = matrix.indptr[row + 1]
                if start == end:
                    continue
                cols = matrix.indices[start:end]
                row_mask = np.zeros(cols.size, dtype=bool)
                for col_start, col_end in zip(col_starts, col_ends):
                    row_mask |= (cols >= col_start) & (cols < col_end)
                mask[start:end] = row_mask
        return mask

    def _apply_nnz_mask(self, matrix: csr_matrix, mask: np.ndarray) -> csr_matrix:
        matrix = matrix.tocsr()
        new_indptr = np.empty(matrix.shape[0] + 1, dtype=np.int64)
        new_indptr[0] = 0
        for row in range(matrix.shape[0]):
            new_indptr[row + 1] = new_indptr[row] + int(mask[matrix.indptr[row]:matrix.indptr[row + 1]].sum())
        return csr_matrix((matrix.data[mask], matrix.indices[mask], new_indptr), shape=matrix.shape)

    def _replace_near_blocks_sparse(
        self,
        weak_matrix: csr_matrix,
        pred_matrix: csr_matrix,
        r_vec: Tuple[int, int, int],
        index_dict: Dict[int, List[int]],
        r_cut: float,
    ) -> csr_matrix:
        pairs_by_atom = self._query_near_atom_pairs(r_vec, r_cut)
        pred_inside_mask = self._block_membership_mask(pred_matrix, index_dict, pairs_by_atom)
        weak_inside_mask = self._block_membership_mask(weak_matrix, index_dict, pairs_by_atom)
        weak_far = self._apply_nnz_mask(weak_matrix, ~weak_inside_mask)
        pred_near = self._apply_nnz_mask(pred_matrix, pred_inside_mask)
        merged = weak_far + pred_near
        merged.sum_duplicates()
        merged.eliminate_zeros()
        return merged

    def _count_csr_r_vectors(self, path: Union[str, Path]) -> Tuple[int, int]:
        with open(path, 'r') as fh:
            return self._read_csr_text_header(fh)
    
    @time_execution
    def add_matrxi_element(self):
        index_dict = self.read_stru()
        weak_csr = self._ensure_weak_hr_csr()
        basis_num, weak_r_num = self._count_csr_r_vectors(weak_csr)
        pred_basis_num, pred_r_num = self._count_csr_r_vectors(self.hr1)
        if basis_num != pred_basis_num:
            raise ValueError(f"Basis dimension mismatch: weak={basis_num}, predicted={pred_basis_num}")
        estimated_dense_gib = self._estimated_dense_gib(max(weak_r_num, pred_r_num))
        print(
            f"Using sparse dense-completion path. Equivalent dense memory would be "
            f"{estimated_dense_gib:.1f} GiB, so matrices are processed one R block at a time.",
            flush=True,
        )

        r_cut = 7
        hr_add_file = os.path.join(self.save_path, 'predict_hr_tot')
        pred_iter = self._iter_csr_text(self.hr1)
        current_pred = next(pred_iter, None)
        completed = 0
        replaced = 0

        with open(hr_add_file, 'w') as f1:
            f1.write('STEP: 0\n')
            f1.write(f'Matrix Dimension of H(R): {basis_num}\n')
            f1.write(f'Matrix number of H(R): {weak_r_num}\n')

            for weak_r, weak_matrix, _ in self._iter_csr_text(weak_csr):
                while current_pred is not None and current_pred[0] < weak_r:
                    current_pred = next(pred_iter, None)

                if current_pred is not None and current_pred[0] == weak_r:
                    pred_matrix = current_pred[1]
                    out_matrix = self._replace_near_blocks_sparse(weak_matrix, pred_matrix, weak_r, index_dict, r_cut)
                    current_pred = next(pred_iter, None)
                    replaced += 1
                else:
                    out_matrix = weak_matrix

                self._write_csr_text_matrix(f1, weak_r, out_matrix)
                completed += 1
                print(f"Step B sparse completion: {completed}/{weak_r_num} R blocks written, replaced={replaced}", flush=True)

        print(f"Sparse dense-completion Hamiltonian saved to: {hr_add_file}", flush=True)
            

def main():
    pass

if __name__ == "__main__":
    main()



# # mp-22120

# init_path = '/home/zujiandai/file-test/24.12.5-HCP/test/25.4.9-test-band/mp-559437'
# data_path = '/home/zujiandai/file-test/24.12.5-HCP/soc_data_collect/soc-data1/mp-559437'
# save_path = init_path
# hr1 = os.path.join(init_path, 'precise_H_pred')
# # hr1 = '/home/zujiandai/file-test/24.12.5-HCP/test/25.4.11-test-rcut-band/soc-data1/mp-10030/hr_cut.csr'
# hr2 = os.path.join(data_path, 'data-HR-sparse_SPIN0_weak.csr')
# sr2 = os.path.join(data_path, 'data-SR-sparse_SPIN0.csr')
# nspin = 4
# stru_name = 'mp-559437'

# my_class = add_hs_matrix(nspin, stru_name, hr1, hr2, sr2, save_path)
# stru_path = my_class.search_stru()
# # print(stru_path)
# my_class.read_stru()
# my_class.add_matrxi_element()

# path_data = '/home/zujiandai/file-test/24.12.5-HCP/test/25.4.16-test-band'
# for filename in os.listdir(path_data):
    
#     # if filename.startswith('mp-'):
#     if filename == 'mp-1001615':
#         stru_name = filename
#         print(stru_name)
#         path_data_now = os.path.join(path_data, stru_name)
#         save_path = path_data_now 
#         hr1 = os.path.join(path_data_now, 'precise_H')
#         hr2 = None
#         sr2 = None
#         nspin = 4
#         # 第一次调用类，获取初始结构文件信息
#         my_class = add_hs_matrix(nspin, stru_name, hr1, hr2, sr2, save_path)
#         stru_path = my_class.search_stru()
#         path_origin = stru_path .rsplit('/', 1)[0]
#         # 第二次调用类，正常进行计算
#         hr2 = os.path.join(path_origin, 'data-HR-sparse_SPIN0_strong.csr')
#         sr2 = os.path.join(path_origin, 'data-SR-sparse_SPIN0.csr')
#         my_class = add_hs_matrix(nspin, stru_name, hr1, hr2, sr2, save_path)
#         stru_path = my_class.search_stru()
#         my_class.read_stru()
#         my_class.add_matrxi_element()
#         print(stru_name,'补充哈密顿量矩阵元完成')
        
