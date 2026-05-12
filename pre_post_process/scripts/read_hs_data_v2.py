from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path

from scipy.sparse import csr_matrix
from scipy.linalg import block_diag

from ase.io import read, write
from ase.neighborlist import NeighborList

import numpy as np
import os
import re
import sys
import time
import torch

try:
    from timing_utils import time_execution
except ImportError:
    # If timing_utils is not found (e.g. running script directly), try adding parent dir
    sys.path.append(str(Path(__file__).parent.parent))
    try:
        from timing_utils import time_execution
    except ImportError:
        # Fallback decorator if import fails
        def time_execution(func):
            return func

class HamiltonianDataReader:
    def __init__(
        self,
        usage: str,
        nspin: int,
        out_path: Union[str, Path],
        stru_file: Union[str, Path],
        weak_file: Union[str, Path],
        overlap_file: Union[str, Path],
        label_file: Optional[Union[str, Path]] = None,  # 默认 None
    ):
        self.usage = usage
        self.nspin = nspin
        self.out_path = Path(out_path)
        self.stru_file = Path(stru_file)
        self.weak_file = Path(weak_file)
        self.overlap_file = Path(overlap_file)
        if self.usage == "inference":
            self.label_file = None
        else:
            self.label_file = label_file

        self.orb_origin = {'H': 5,   'He': 5,  'Li': 7,  'Be': 7,  'B': 13,  'C': 13,  'N': 13,  'O': 13,  'F': 13, 'Ne': 13, 
                           'Na': 15, 'Mg': 15, 'Al': 13, 'Si': 13, 'P': 13,  'S': 13,  'Cl': 13, 'Ar': 13, 'K': 15, 
                           'Sc': 27, 'V': 27,  'Fe': 27, 'Co': 27, 'Ni': 27, 'Cu': 27, 'Zn': 27, 'Ga': 25, 'Ge': 25, 
                           'Br': 13, 'Y': 27,  'Nb': 27, 'Mo': 27, 'Pd': 25, 'Ag': 27, 'Cd': 27, 'In': 25, 'Sn': 25, 
                           'Sb': 25, 'Te': 25, 'I': 13,  'Xe': 13, 'Hf': 27, 'Ta': 27, 'Re': 27, 'Pt': 27, 'Au': 27, 
                           'Hg': 27, 'Tl': 25, 'Pb': 25, 'Bi': 25, 'Ca': 15, 'Ti': 27, 'Cr': 27, 'Mn': 27, 'Kr': 13, 
                           'Rb': 15, 'Sr': 15, 'Zr': 27, 'Tc': 27, 'Ru': 27, 'Rh': 27, 'Cs': 15, 'Ba': 15, 'W': 27, 
                           'Os': 27, 'Ir': 27, 'As': 13, 'Se': 13}

    # 用 ASE NeighborList 搜索近邻，并按 R=(Rx,Ry,Rz) 分组。返回: pairs_by_R[(Rx,Ry,Rz)] = [(ii, jj, distance_vec(3,)), ...]
    # NeighborList 判断条件是 d < cutoff_i + cutoff_j，因此每个原子 cutoff 取 r_cut/2
    # self_interaction=True: 保留 ii==jj (含 R!=0 的自相互项)

    def build_neighbor_pairs_by_R(self, atoms, r_cut: float,) -> Dict[Tuple[int, int, int], List[Tuple[int, int, np.ndarray]]]:
        n = len(atoms)
        cutoffs = [r_cut * 0.5] * n
        nl = NeighborList(cutoffs, skin=0.0, bothways=True, self_interaction=True)
        nl.update(atoms)

        cell = atoms.cell
        pos = atoms.positions
        pairs_by_R: Dict[Tuple[int, int, int], List[Tuple[int, int, np.ndarray]]] = {}
        for ii in range(n):
            neigh_idx, neigh_offsets = nl.get_neighbors(ii)  # offsets: integer shift vectors
            for jj, off in zip(neigh_idx, neigh_offsets):
                R = (int(off[0]), int(off[1]), int(off[2]))
                # distance 定义：R*cell + rj - ri
                distance_vec = off[0] * cell[0] + off[1] * cell[1] + off[2] * cell[2] + pos[jj] - pos[ii]
                if np.linalg.norm(distance_vec) < r_cut:
                    pairs_by_R.setdefault(R, []).append((int(ii), int(jj), np.array(distance_vec, dtype=float)))
        for R in pairs_by_R:
            pairs_by_R[R].sort(key=lambda x: (x[0], x[1]))
        return pairs_by_R

    def transform_orb(self, nspin):
        self.abacus2deeph = {}
        self.abacus2deeph[0] = np.eye(1)
        self.abacus2deeph[1] = np.eye(3)[[1, 2, 0]]
        self.abacus2deeph[2] = np.eye(5)[[0, 3, 4, 1, 2]]
        self.abacus2deeph[3] = np.eye(7)[[0, 1, 2, 3, 4, 5, 6]]
        minus_dict = {
            1: [0, 1],
            2: [3, 4],
            3: [1, 2, 5, 6],
        }
        # 应用符号翻转
        for k, v in minus_dict.items():
            self.abacus2deeph[k][v] *= -1

        # 根据 nspin 决定轨道列表
        if nspin == 1:
            orb_list = [0, 0, 0, 0, 1, 1, 2, 2, 3]
        elif nspin == 4:
            orb_list = [0, 0, 0, 0, 1, 1, 2, 2, 3, 0, 0, 0, 0, 1, 1, 2, 2, 3]
        else:
            raise ValueError(f"Unsupported nspin value: {nspin}")

        # 修正调用方式，使用 [] 访问字典
        transofrm_matrix = block_diag(*[self.abacus2deeph[l_number] for l_number in orb_list])
        return transofrm_matrix
    @time_execution
    def read_stru(self):
        stru_file = self.stru_file
        # 读取 STRU 文件
        self.atoms = read(stru_file, format="abacus")
        # 计算轨道数目总和，对于自旋为4情况，矩阵维度进行扩充
        elements = self.atoms.get_chemical_symbols()
        self.stru_dim = sum(self.orb_origin.get(element, 0) for element in elements)
        if self.nspin == 4:
            self.stru_dim = 2 * self.stru_dim
        # 对于每一个原子指标 i 生成轨道 patching 字典[1*27] 维度列表
        # 如果考虑 自旋轨道耦合情况，轨道指标会翻倍，每一个原子指标 i 生成轨道 patching 字典[1*54] 维度列表
        # 所有补零矩阵都是按照 4s2p2d1f 形式进行的
        count_element = 0
        count_matrix_dim = -1
        self.index_relation = {}
        for element in elements:
            if self.orb_origin.get(element) == 5:    # 2s1p
                orital_patch0 = [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            elif self.orb_origin.get(element) == 7:  # 4s1p
                orital_patch0 = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            elif self.orb_origin.get(element) == 13: # 2s2p1d
                orital_patch0 = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            elif self.orb_origin.get(element) == 15: # 4s2p1d
                orital_patch0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
            elif self.orb_origin.get(element) == 25: # 2s2p2d1f
                orital_patch0 = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
            elif self.orb_origin.get(element) == 27: # 4s2p2d1f
                orital_patch0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]
            else:
                raise ValueError(f"Unknown element orbital setting: {element}")
            # 对自旋等于四轨道进行处理  
            if self.nspin != 4:
                orital_patch = orital_patch0
            else:
                orital_patch = []
                for temp_value in orital_patch0:
                    orital_patch.append(temp_value)
                    orital_patch.append(temp_value)

            orbital_index = []
            for temp_value in orital_patch:
                count_matrix_dim = count_matrix_dim + temp_value
                orbital_index.append(count_matrix_dim)
            self.index_relation[count_element] = [element, orital_patch, orbital_index]
            count_element = count_element + 1
    @time_execution
    def read_data(self):
        # inference: strong_label_path 直接指向 weak_file（保持你原逻辑）
        if self.usage == "inference":
            strong_label_path = self.weak_file
        elif self.usage == "train":
            strong_label_path = self.label_file
        else:
            raise ValueError(f"Unsupported usage: {self.usage}")
        weak_label_path = self.weak_file
        s_matrix_path = self.overlap_file
        # 先读取精标签数据，最后读取弱标签数据
        # 创建两个列表，一个存储指标信息[Rx, Ry, Rz, i, j, dx, dy, dz] 1*8 矩阵形式
        # 另一个为4*27*27 维度矩阵，分别存储不同pair对的精、弱标签、交叠矩阵和hamiltion和patching矩阵

        self.index_list = []
        self.ele_list = []
        self.matrix_list = []
        self.unify_orb_num = 27
        if self.nspin == 4:
            self.unify_orb_num = 2 * self.unify_orb_num

        # ====== 预先读出 strong/weak/overlap 的 R 列表 ======
        with open(strong_label_path, "r") as fread_strong:
            fread_strong.readline()
            line_strong = fread_strong.readline()
            basis_num = int(line_strong.split()[-1])
            if basis_num != self.stru_dim:
                print(f" STRU 结构计算矩阵维度和{strong_label_path} 矩阵维度不一致，请检查结果 ")
                sys.exit(2)  # 退出程序，返回错误代码 2
            line_strong = fread_strong.readline()
            R_num_strong = int(line_strong.split()[-1])
            R_direct_coor_strong = np.zeros([R_num_strong, 3], dtype=int)
            for iR in range(R_num_strong):
                # 获取 R 指标
                line = fread_strong.readline().split()
                R_direct_coor_strong[iR, 0] = int(line[0])
                R_direct_coor_strong[iR, 1] = int(line[1])
                R_direct_coor_strong[iR, 2] = int(line[2])
                data_size = int(line[3])
                if data_size != 0:
                    fread_strong.readline()
                    fread_strong.readline()
                    fread_strong.readline()

        # 读取弱标签数据数据前三行信息
        with open(weak_label_path, "r") as fread_weak:
            fread_weak.readline()
            line_weak = fread_weak.readline()
            basis_num = int(line_weak.split()[-1])
            if basis_num != self.stru_dim:
                print(f" STRU 结构计算矩阵维度和{weak_label_path} 矩阵维度不一致，请检查结果 ")
                sys.exit(2)
            line_weak = fread_weak.readline()
            R_num_weak = int(line_weak.split()[-1])
            R_direct_coor_weak = np.zeros([R_num_weak, 3], dtype=int)
            for iR in range(R_num_weak):
                line = fread_weak.readline().split()
                R_direct_coor_weak[iR, 0] = int(line[0])
                R_direct_coor_weak[iR, 1] = int(line[1])
                R_direct_coor_weak[iR, 2] = int(line[2])
                data_size = int(line[3])
                if data_size != 0:
                    fread_weak.readline()
                    fread_weak.readline()
                    fread_weak.readline()

        # 读取交叠数据数据前三行信息
        with open(s_matrix_path, "r") as fread_overlap:
            fread_overlap.readline()
            line_overlap = fread_overlap.readline()
            basis_num = int(line_overlap.split()[-1])
            if basis_num != self.stru_dim:
                print(f" STRU 结构计算矩阵维度和{s_matrix_path} 矩阵维度不一致，请检查结果 ")
                sys.exit(2)
            line_overlap = fread_overlap.readline()
            R_num_overlap = int(line_overlap.split()[-1])
            R_direct_coor_overlap = np.zeros([R_num_overlap, 3], dtype=int)
            for iR in range(R_num_overlap):
                line = fread_overlap.readline().split()
                R_direct_coor_overlap[iR, 0] = int(line[0])
                R_direct_coor_overlap[iR, 1] = int(line[1])
                R_direct_coor_overlap[iR, 2] = int(line[2])
                data_size = int(line[3])
                if data_size != 0:
                    fread_overlap.readline()
                    fread_overlap.readline()
                    fread_overlap.readline()

        # ====== strong/weak 共有的 R 列表 ======
        R_coor_both = []
        for iR_strong in range(R_num_strong):
            for iR_weak in range(R_num_weak):
                if (
                    R_direct_coor_strong[iR_strong, 0] == R_direct_coor_weak[iR_weak, 0]
                    and R_direct_coor_strong[iR_strong, 1] == R_direct_coor_weak[iR_weak, 1]
                    and R_direct_coor_strong[iR_strong, 2] == R_direct_coor_weak[iR_weak, 2]
                ):
                    R_coor_both.append(R_direct_coor_strong[iR_strong])

        # ====== 利用NeighborList 近邻分组（外接函数） ======
        R_cut = 8.0
        pairs_by_R = self.build_neighbor_pairs_by_R(self.atoms, R_cut)
        # 只保留 R_coor_both 里出现的 R，且按 R_coor_both 的顺序处理（不改变你的逻辑顺序）
        R_set_both = { (int(r[0]), int(r[1]), int(r[2])) for r in R_coor_both }

        # ====== 按照 R_coor_both 顺序，同时读取 strong/weak/overlap 并处理该 R 下的 pair ======
        with open(strong_label_path, "r") as fread_strong, open(weak_label_path, "r") as fread_weak, open(s_matrix_path, "r") as fread_overlap:
            # 跳过前三行
            fread_strong.readline()
            fread_strong.readline()
            fread_strong.readline()
            fread_weak.readline()
            fread_weak.readline()
            fread_weak.readline()
            fread_overlap.readline()
            fread_overlap.readline()
            fread_overlap.readline()

            for iR in range(len(R_coor_both)):
                R_target = (int(R_coor_both[iR][0]), int(R_coor_both[iR][1]), int(R_coor_both[iR][2]))
                #print(f"iR 指标为{R_coor_both[iR]}", flush=True)

                # 如果该 R 在 cutoff 下根本没有任何近邻对，直接把文件读指针推进到该 R block，读取矩阵但不生成样本也可以；
                # 为了尽量不改你的逻辑，这里仍然读取矩阵，但不会进入 pair 循环（生成样本为0）。
                pair_list = pairs_by_R.get(R_target, [])

                # ===== strong: 找到对应 R 的 block 并读出矩阵 =====
                R_stong_temp = np.zeros([3], dtype=int)
                line_strong = fread_strong.readline().split()
                R_stong_temp[:] = [int(line_strong[0]), int(line_strong[1]), int(line_strong[2])]
                data_size = int(line_strong[3])

                while True:
                    if (R_stong_temp[0], R_stong_temp[1], R_stong_temp[2]) == R_target:
                        break
                    else:
                        if data_size != 0:
                            fread_strong.readline()
                            fread_strong.readline()
                            fread_strong.readline()
                        line_strong = fread_strong.readline().split()
                        R_stong_temp[:] = [int(line_strong[0]), int(line_strong[1]), int(line_strong[2])]
                        data_size = int(line_strong[3])

                if self.nspin != 4:
                    matrix_R_strong = np.zeros([basis_num, basis_num], dtype=float)
                else:
                    matrix_R_strong = np.zeros([basis_num, basis_num], dtype=complex)

                if data_size != 0:
                    if self.nspin != 4:
                        data = np.zeros((data_size,), dtype=float)
                        line_vals = fread_strong.readline().split()
                        for index in range(data_size):
                            data[index] = float(line_vals[index])
                    else:
                        data = np.zeros((data_size,), dtype=complex)
                        line_vals = re.findall(r"[(](.*?)[)]", fread_strong.readline())
                        for index in range(data_size):
                            value = line_vals[index].split(",")
                            data[index] = complex(float(value[0]), float(value[1]))

                    indices = np.zeros((data_size,), dtype=int)
                    indptr = np.zeros((basis_num + 1,), dtype=int)

                    line_idx = fread_strong.readline().split()
                    for index in range(data_size):
                        indices[index] = int(line_idx[index])

                    line_ptr = fread_strong.readline().split()
                    for index in range(basis_num + 1):
                        indptr[index] = int(line_ptr[index])

                    matrix_R_strong = csr_matrix((data, indices, indptr), shape=(basis_num, basis_num)).toarray()

                # ===== weak: 找到对应 R 的 block 并读出矩阵 =====
                R_weak_temp = np.zeros([3], dtype=int)
                line_weak = fread_weak.readline().split()
                R_weak_temp[:] = [int(line_weak[0]), int(line_weak[1]), int(line_weak[2])]
                data_size = int(line_weak[3])

                while True:
                    if (R_weak_temp[0], R_weak_temp[1], R_weak_temp[2]) == R_target:
                        break
                    else:
                        if data_size != 0:
                            fread_weak.readline()
                            fread_weak.readline()
                            fread_weak.readline()
                        line_weak = fread_weak.readline().split()
                        R_weak_temp[:] = [int(line_weak[0]), int(line_weak[1]), int(line_weak[2])]
                        data_size = int(line_weak[3])

                if self.nspin != 4:
                    matrix_R_weak = np.zeros([basis_num, basis_num], dtype=float)
                else:
                    matrix_R_weak = np.zeros([basis_num, basis_num], dtype=complex)

                if data_size != 0:
                    if self.nspin != 4:
                        data = np.zeros((data_size,), dtype=float)
                        line_vals = fread_weak.readline().split()
                        for index in range(data_size):
                            data[index] = float(line_vals[index])
                    else:
                        data = np.zeros((data_size,), dtype=complex)
                        line_vals = re.findall(r"[(](.*?)[)]", fread_weak.readline())
                        for index in range(data_size):
                            value = line_vals[index].split(",")
                            data[index] = complex(float(value[0]), float(value[1]))

                    indices = np.zeros((data_size,), dtype=int)
                    indptr = np.zeros((basis_num + 1,), dtype=int)

                    line_idx = fread_weak.readline().split()
                    for index in range(data_size):
                        indices[index] = int(line_idx[index])

                    line_ptr = fread_weak.readline().split()
                    for index in range(basis_num + 1):
                        indptr[index] = int(line_ptr[index])

                    matrix_R_weak = csr_matrix((data, indices, indptr), shape=(basis_num, basis_num)).toarray()

                # ===== overlap: 找到对应 R 的 block 并读出矩阵 =====
                R_overlap_temp = np.zeros([3], dtype=int)
                line_overlap = fread_overlap.readline().split()
                R_overlap_temp[:] = [int(line_overlap[0]), int(line_overlap[1]), int(line_overlap[2])]
                data_size = int(line_overlap[3])

                while True:
                    if (R_overlap_temp[0], R_overlap_temp[1], R_overlap_temp[2]) == R_target:
                        break
                    else:
                        if data_size != 0:
                            fread_overlap.readline()
                            fread_overlap.readline()
                            fread_overlap.readline()
                        line_overlap = fread_overlap.readline().split()
                        R_overlap_temp[:] = [int(line_overlap[0]), int(line_overlap[1]), int(line_overlap[2])]
                        data_size = int(line_overlap[3])

                if self.nspin != 4:
                    matrix_R_overlap = np.zeros([basis_num, basis_num], dtype=float)
                else:
                    matrix_R_overlap = np.zeros([basis_num, basis_num], dtype=complex)

                if data_size != 0:
                    if self.nspin != 4:
                        data = np.zeros((data_size,), dtype=float)
                        line_vals = fread_overlap.readline().split()
                        for index in range(data_size):
                            data[index] = float(line_vals[index])
                    else:
                        data = np.zeros((data_size,), dtype=complex)
                        line_vals = re.findall(r"[(](.*?)[)]", fread_overlap.readline())
                        for index in range(data_size):
                            value = line_vals[index].split(",")
                            data[index] = complex(float(value[0]), float(value[1]))

                    indices = np.zeros((data_size,), dtype=int)
                    indptr = np.zeros((basis_num + 1,), dtype=int)

                    line_idx = fread_overlap.readline().split()
                    for index in range(data_size):
                        indices[index] = int(line_idx[index])

                    line_ptr = fread_overlap.readline().split()
                    for index in range(basis_num + 1):
                        indptr[index] = int(line_ptr[index])

                    matrix_R_overlap = csr_matrix((data, indices, indptr), shape=(basis_num, basis_num)).toarray()

                # ===== 利用近邻搜索方法只遍历该 R 下的近邻 pair =====
                for (ii, jj, distance_vec) in pair_list:
                    # 创建列表指标矩阵，并且添加到 self.index_list 中
                    temp_label = np.zeros([8], dtype=float)
                    if self.nspin != 4:
                        temp_data = np.zeros([4, self.unify_orb_num, self.unify_orb_num], dtype=float)
                    else:
                        temp_data = np.zeros([4, self.unify_orb_num, self.unify_orb_num], dtype=complex)

                    temp_label[0:3] = np.array(R_target, dtype=float)
                    temp_label[3] = float(ii)
                    temp_label[4] = float(jj)
                    temp_label[5:8] = distance_vec

                    self.index_list.append(temp_label)
                    self.ele_list.append((self.atoms.get_chemical_symbols()[ii], self.atoms.get_chemical_symbols()[jj]))

                    # ===== 27×27(或54×54) 用 outer + ix_ 向量化 =====
                    patch_i = np.asarray(self.index_relation[ii][1], dtype=float)  # (unify_orb_num,)
                    patch_j = np.asarray(self.index_relation[jj][1], dtype=float)
                    mask = np.outer(patch_i, patch_j)  # (unify, unify)

                    idx_i = np.asarray(self.index_relation[ii][2], dtype=int)
                    idx_j = np.asarray(self.index_relation[jj][2], dtype=int)

                    # 取子块并乘 mask
                    sub_strong = matrix_R_strong[np.ix_(idx_i, idx_j)]
                    sub_weak = matrix_R_weak[np.ix_(idx_i, idx_j)]
                    sub_overlap = matrix_R_overlap[np.ix_(idx_i, idx_j)]

                    temp_data[0] = sub_strong * mask
                    temp_data[1] = sub_weak * mask
                    temp_data[2] = sub_overlap * mask
                    temp_data[3] = mask

                    # ===== 原来的轨道变换逻辑不变 =====
                    if self.nspin == 1:
                        T = self.transform_orb(1)
                        temp_data[0] = T @ temp_data[0] @ T.T
                        temp_data[1] = T @ temp_data[1] @ T.T
                        temp_data[2] = T @ temp_data[2] @ T.T

                    elif self.nspin == 4:
                        # 你的原逻辑：先 reshape(27,2,27,2) -> transpose -> reshape(54,54) 再旋转
                        T = self.transform_orb(4)

                        reshaped_data_0 = temp_data[0].reshape((27, 2, 27, 2))
                        reshaped_data_0 = reshaped_data_0.transpose((1, 0, 3, 2)).reshape((2 * 27, 2 * 27))
                        temp_data[0] = T @ reshaped_data_0 @ T.T

                        reshaped_data_1 = temp_data[1].reshape((27, 2, 27, 2))
                        reshaped_data_1 = reshaped_data_1.transpose((1, 0, 3, 2)).reshape((2 * 27, 2 * 27))
                        temp_data[1] = T @ reshaped_data_1 @ T.T

                        reshaped_data_2 = temp_data[2].reshape((27, 2, 27, 2))
                        reshaped_data_2 = reshaped_data_2.transpose((1, 0, 3, 2)).reshape((2 * 27, 2 * 27))
                        temp_data[2] = T @ reshaped_data_2 @ T.T

                    self.matrix_list.append(temp_data)

        # ===== 保存张量 =====
        arr = np.array(self.matrix_list)
        self.matrix_tensor = torch.from_numpy(arr).to(torch.complex64)

        label_tensor = self.matrix_tensor[:, 0] - self.matrix_tensor[:, 1]
        descriptor_tensor = self.matrix_tensor[:, 1]

        if self.usage == "inference":
            overlap_tensor = None
        elif self.usage == "train":
            overlap_tensor = self.matrix_tensor[:, 2]
        else:
            overlap_tensor = None

        mask_tensor = self.matrix_tensor[:, 3]

        arr2 = np.array(self.index_list)
        self.index_tensor = torch.from_numpy(arr2).to(torch.complex64)

        edge_vec = self.index_tensor[:, 5:8].real.to(torch.float32)
        edge_src = self.index_tensor[:, 3].real.to(torch.long)
        edge_dst = self.index_tensor[:, 4].real.to(torch.long)

        input_path = os.path.join(self.out_path, "input_inference.pth")
        output_path = os.path.join(self.out_path, "output_inference.pth")

        self.data = [descriptor_tensor, overlap_tensor, mask_tensor, edge_vec, edge_src, edge_dst, self.ele_list, output_path]

        if self.usage == "inference":
            self.label = None
        elif self.usage == "train":
            self.label = [label_tensor]
        else:
            self.label = None

        sample = (self.data, self.label)
        torch.save(sample, input_path)
        return sample


def main():
    pass


if __name__ == "__main__":
    main()
