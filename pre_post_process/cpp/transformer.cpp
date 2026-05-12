#include "transformer.h"
#include <omp.h>
#include <iostream>

TransformationEngine::TransformationEngine(int nspin) : nspin_(nspin) {
    unify_orb_num_ = 27;
    if (nspin_ == 4) unify_orb_num_ *= 2;

    orb_origin_ = {
        {"H", 5}, {"He", 5}, {"Li", 7}, {"Be", 7}, {"B", 13}, {"C", 13}, {"N", 13}, {"O", 13},
        {"F", 13}, {"Ne", 13}, {"Na", 15}, {"Mg", 15}, {"Al", 13}, {"Si", 13}, {"P", 13},
        {"S", 13}, {"Cl", 13}, {"Ar", 13}, {"K", 15}, {"Sc", 27}, {"V", 27}, {"Fe", 27},
        {"Co", 27}, {"Ni", 27}, {"Cu", 27}, {"Zn", 27}, {"Ga", 25}, {"Ge", 25}, {"Br", 13},
        {"Y", 27}, {"Nb", 27}, {"Mo", 27}, {"Pd", 25}, {"Ag", 27}, {"Cd", 27}, {"In", 25},
        {"Sn", 25}, {"Sb", 25}, {"Te", 25}, {"I", 13}, {"Xe", 13}, {"Hf", 27}, {"Ta", 27},
        {"Re", 27}, {"Pt", 27}, {"Au", 27}, {"Hg", 27}, {"Tl", 25}, {"Pb", 25}, {"Bi", 25},
        {"Ca", 15}, {"Ti", 27}, {"Cr", 27}, {"Mn", 27}, {"Kr", 13}, {"Rb", 15}, {"Sr", 15},
        {"Zr", 27}, {"Tc", 27}, {"Ru", 27}, {"Rh", 27}, {"Cs", 15}, {"Ba", 15}, {"W", 27},
        {"Os", 27}, {"Ir", 27}, {"As", 13}, {"Se", 13}
    };
    
    trans_mat_ = get_transform_matrix();
}

std::vector<int> TransformationEngine::get_orbital_patch_pattern(int orb_type) {
    std::vector<int> p(27, 0);
    if (orb_type == 5) {
        p[0] = p[1] = p[4] = p[5] = p[6] = 1;
    } else if (orb_type == 7) {
        for (int i = 0; i < 7; ++i) p[i] = 1;
    } else if (orb_type == 13) {
        for (int i = 0; i < 15; ++i) {
            if (i != 2 && i != 3) p[i] = 1;
        }
    } else if (orb_type == 15) {
        for (int i = 0; i < 15; ++i) p[i] = 1;
    } else if (orb_type == 25) {
        for (int i = 0; i < 27; ++i) {
            if (i != 2 && i != 3) p[i] = 1;
        }
    } else if (orb_type == 27) {
        for (int i = 0; i < 27; ++i) p[i] = 1;
    }
    return p;
}

Eigen::MatrixXcd TransformationEngine::get_transform_matrix() {
    std::vector<int> orb_list;
    if (nspin_ == 1) {
        orb_list = {0, 0, 0, 0, 1, 1, 2, 2, 3};
    } else {
        orb_list = {0, 0, 0, 0, 1, 1, 2, 2, 3, 0, 0, 0, 0, 1, 1, 2, 2, 3};
    }

    std::vector<Eigen::MatrixXd> abacus2deeph(4);
    abacus2deeph[0] = Eigen::MatrixXd::Identity(1, 1);
    
    abacus2deeph[1] = Eigen::MatrixXd::Zero(3, 3);
    abacus2deeph[1](0, 1) = 1; abacus2deeph[1](1, 2) = 1; abacus2deeph[1](2, 0) = 1;
    abacus2deeph[1](0, 1) *= -1; abacus2deeph[1](1, 2) *= -1; // minus_dict
    
    abacus2deeph[2] = Eigen::MatrixXd::Zero(5, 5);
    abacus2deeph[2](0, 0) = 1; abacus2deeph[2](1, 3) = 1; abacus2deeph[2](2, 4) = 1;
    abacus2deeph[2](3, 1) = 1; abacus2deeph[2](4, 2) = 1;
    abacus2deeph[2](3, 1) *= -1; abacus2deeph[2](4, 2) *= -1;

    abacus2deeph[3] = Eigen::MatrixXd::Identity(7, 7);
    abacus2deeph[3](1, 1) *= -1; abacus2deeph[3](2, 2) *= -1; 
    abacus2deeph[3](5, 5) *= -1; abacus2deeph[3](6, 6) *= -1;

    int total_dim = 0;
    for (int l : orb_list) {
        total_dim += (l == 0 ? 1 : (l == 1 ? 3 : (l == 2 ? 5 : 7)));
    }

    Eigen::MatrixXcd T = Eigen::MatrixXcd::Zero(total_dim, total_dim);
    int curr = 0;
    for (int l : orb_list) {
        int d = (l == 0 ? 1 : (l == 1 ? 3 : (l == 2 ? 5 : 7)));
        T.block(curr, curr, d, d) = abacus2deeph[l].cast<std::complex<double>>();
        curr += d;
    }
    return T;
}

TransformationEngine::ProcessedData TransformationEngine::process(const StruData& stru, const CSRData& csr, const std::vector<Edge>& edges) {
    int n_atoms = stru.atoms.size();
    std::vector<AtomInfo> atom_infos(n_atoms);
    int count_matrix_dim = -1;
    
    for (int i = 0; i < n_atoms; ++i) {
        int orb_type = orb_origin_[stru.atoms[i].element];
        std::vector<int> base_patch = get_orbital_patch_pattern(orb_type);
        
        atom_infos[i].element = stru.atoms[i].element;
        if (nspin_ != 4) {
            atom_infos[i].orbital_patch = base_patch;
        } else {
            for (int val : base_patch) {
                atom_infos[i].orbital_patch.push_back(val);
                atom_infos[i].orbital_patch.push_back(val);
            }
        }
        
        for (int val : atom_infos[i].orbital_patch) {
            count_matrix_dim += val;
            atom_infos[i].orbital_index.push_back(count_matrix_dim);
        }
    }

    int n_edges = edges.size();
    ProcessedData result;
    result.descriptor = torch::zeros({n_edges, unify_orb_num_, unify_orb_num_}, torch::kComplexFloat);
    result.mask = torch::zeros({n_edges, unify_orb_num_, unify_orb_num_}, torch::kFloat32);
    result.edge_vec = torch::zeros({n_edges, 3}, torch::kFloat32);
    result.edge_src = torch::zeros({n_edges}, torch::kLong);
    result.edge_dst = torch::zeros({n_edges}, torch::kLong);
    result.ele_list.resize(n_edges);

    #pragma omp parallel for
    for (int e = 0; e < n_edges; ++e) {
        const auto& edge = edges[e];
        RVector r = {edge.r_offset(0), edge.r_offset(1), edge.r_offset(2)};
        
        int ii = edge.atom_i;
        int jj = edge.atom_j;
        
        const auto& info_i = atom_infos[ii];
        const auto& info_j = atom_infos[jj];
        
        Eigen::MatrixXf mask_f = Eigen::MatrixXf::Zero(unify_orb_num_, unify_orb_num_);
        for (int row = 0; row < unify_orb_num_; ++row) {
            for (int col = 0; col < unify_orb_num_; ++col) {
                if (info_i.orbital_patch[row] && info_j.orbital_patch[col]) {
                    mask_f(row, col) = 1.0f;
                }
            }
        }

        Eigen::MatrixXcd sub_w = Eigen::MatrixXcd::Zero(unify_orb_num_, unify_orb_num_);
        auto it = csr.matrices.find(r);
        int found_count = 0;
        if (it != csr.matrices.end()) {
            const auto& mat = it->second;
            // Extraction from CSR
            for (int row_local = 0; row_local < (int)info_i.orbital_index.size(); ++row_local) {
                if (!info_i.orbital_patch[row_local]) continue;
                int row_global = info_i.orbital_index[row_local];
                if (row_global < 0 || row_global >= (int)mat.indptr.size() - 1) continue;
                
                for (int idx_ptr = mat.indptr[row_global]; idx_ptr < mat.indptr[row_global+1]; ++idx_ptr) {
                    int col_global = mat.indices[idx_ptr];
                    
                    // Map col_global back to atom jj
                    for (int col_local = 0; col_local < (int)info_j.orbital_index.size(); ++col_local) {
                        if (info_j.orbital_patch[col_local] && info_j.orbital_index[col_local] == col_global) {
                            if (csr.is_complex) {
                                sub_w(row_local, col_local) = mat.complex_values[idx_ptr];
                            } else {
                                sub_w(row_local, col_local) = mat.values[idx_ptr];
                            }
                            found_count++;
                            break;
                        }
                    }
                }
            }
        }
        //if (e % 100 == 0) std::cout << "Edge " << e << " found " << found_count << " elements." << std::endl;

        // Apply Transformation
        Eigen::MatrixXcd transformed;
        if (nspin_ == 4) {
            // Reshuffle (27, 2, 27, 2) -> (2, 27, 2, 27)
            Eigen::MatrixXcd reshuffled = Eigen::MatrixXcd::Zero(54, 54);
            for (int i = 0; i < 27; ++i) {
                for (int j = 0; j < 27; ++j) {
                    reshuffled(i, j) = sub_w(2 * i, 2 * j);           // up-up
                    reshuffled(i, j + 27) = sub_w(2 * i, 2 * j + 1);    // up-down
                    reshuffled(i + 27, j) = sub_w(2 * i + 1, 2 * j);    // down-up
                    reshuffled(i + 27, j + 27) = sub_w(2 * i + 1, 2 * j + 1); // down-down
                }
            }
            transformed = trans_mat_ * reshuffled * trans_mat_.adjoint();
        } else {
            transformed = trans_mat_ * sub_w * trans_mat_.adjoint();
        }

        // Copy to tensors
        auto desc_ptr = (std::complex<float>*)result.descriptor[e].data_ptr();
        for (int i = 0; i < unify_orb_num_; ++i) {
            for (int j = 0; j < unify_orb_num_; ++j) {
                desc_ptr[i * unify_orb_num_ + j] = std::complex<float>((float)transformed(i, j).real(), (float)transformed(i, j).imag());
            }
        }
        
        auto mask_ptr = (float*)result.mask[e].data_ptr();
        for (int i = 0; i < unify_orb_num_; ++i) {
            for (int j = 0; j < unify_orb_num_; ++j) {
                mask_ptr[i * unify_orb_num_ + j] = mask_f(i, j);
            }
        }

        result.edge_vec[e][0] = (float)edge.dist_vec(0);
        result.edge_vec[e][1] = (float)edge.dist_vec(1);
        result.edge_vec[e][2] = (float)edge.dist_vec(2);
        result.edge_src[e] = edge.atom_i;
        result.edge_dst[e] = edge.atom_j;
        result.ele_list[e] = {info_i.element, info_j.element};
    }
    
    return result;
}

TransformationEngine::ProcessedData TransformationEngine::process(const StruData& stru, const BlockParserResult& blocks, const std::vector<Edge>& edges) {
    int n_atoms = stru.atoms.size();
    std::vector<AtomInfo> atom_infos(n_atoms);
    int count_matrix_dim = -1;
    
    for (int i = 0; i < n_atoms; ++i) {
        int orb_type = orb_origin_[stru.atoms[i].element];
        std::vector<int> base_patch = get_orbital_patch_pattern(orb_type);
        
        atom_infos[i].element = stru.atoms[i].element;
        if (nspin_ != 4) {
            atom_infos[i].orbital_patch = base_patch;
        } else {
            for (int val : base_patch) {
                atom_infos[i].orbital_patch.push_back(val);
                atom_infos[i].orbital_patch.push_back(val);
            }
        }
        
        for (int val : atom_infos[i].orbital_patch) {
            count_matrix_dim += val;
            atom_infos[i].orbital_index.push_back(count_matrix_dim);
        }
    }

    int n_edges = edges.size();
    ProcessedData result;
    result.descriptor = torch::zeros({n_edges, unify_orb_num_, unify_orb_num_}, torch::kComplexFloat);
    result.mask = torch::zeros({n_edges, unify_orb_num_, unify_orb_num_}, torch::kFloat32);
    result.edge_vec = torch::zeros({n_edges, 3}, torch::kFloat32);
    result.edge_src = torch::zeros({n_edges}, torch::kLong);
    result.edge_dst = torch::zeros({n_edges}, torch::kLong);
    result.ele_list.resize(n_edges);

    #pragma omp parallel for
    for (int e = 0; e < n_edges; ++e) {
        const auto& edge = edges[e];
        RVector r = {edge.r_offset(0), edge.r_offset(1), edge.r_offset(2)};
        
        int ii = edge.atom_i;
        int jj = edge.atom_j;
        
        const auto& info_i = atom_infos[ii];
        const auto& info_j = atom_infos[jj];
        
        Eigen::MatrixXf mask_f = Eigen::MatrixXf::Zero(unify_orb_num_, unify_orb_num_);
        for (int row = 0; row < unify_orb_num_; ++row) {
            for (int col = 0; col < unify_orb_num_; ++col) {
                if (info_i.orbital_patch[row] && info_j.orbital_patch[col]) {
                    mask_f(row, col) = 1.0f;
                }
            }
        }

        Eigen::MatrixXcd sub_w = Eigen::MatrixXcd::Zero(unify_orb_num_, unify_orb_num_);
        auto it_r = blocks.matrices.find(r);
        if (it_r != blocks.matrices.end()) {
            auto it_pair = it_r->second.find({ii, jj});
            if (it_pair != it_r->second.end()) {
                for (const auto& block : it_pair->second) {
                    // Map block row_idx and col_idx back to unify_orb_num_ indices
                    for (size_t rb = 0; rb < block.row_idx.size(); ++rb) {
                        int row_global = block.row_idx[rb];
                        // Find local row index for atom ii
                        for (int row_local = 0; row_local < (int)info_i.orbital_index.size(); ++row_local) {
                            if (info_i.orbital_patch[row_local] && info_i.orbital_index[row_local] == row_global) {
                                for (size_t cb = 0; cb < block.col_idx.size(); ++cb) {
                                    int col_global = block.col_idx[cb];
                                    // Find local col index for atom jj
                                    for (int col_local = 0; col_local < (int)info_j.orbital_index.size(); ++col_local) {
                                        if (info_j.orbital_patch[col_local] && info_j.orbital_index[col_local] == col_global) {
                                            if (blocks.is_complex) {
                                                sub_w(row_local, col_local) = block.complex_values[rb * block.col_idx.size() + cb];
                                            } else {
                                                sub_w(row_local, col_local) = block.values[rb * block.col_idx.size() + cb];
                                            }
                                            break;
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Apply Transformation
        Eigen::MatrixXcd transformed;
        if (nspin_ == 4) {
            Eigen::MatrixXcd reshuffled = Eigen::MatrixXcd::Zero(54, 54);
            for (int i = 0; i < 27; ++i) {
                for (int j = 0; j < 27; ++j) {
                    reshuffled(i, j) = sub_w(2 * i, 2 * j);
                    reshuffled(i, j + 27) = sub_w(2 * i, 2 * j + 1);
                    reshuffled(i + 27, j) = sub_w(2 * i + 1, 2 * j);
                    reshuffled(i + 27, j + 27) = sub_w(2 * i + 1, 2 * j + 1);
                }
            }
            transformed = trans_mat_ * reshuffled * trans_mat_.adjoint();
        } else {
            transformed = trans_mat_ * sub_w * trans_mat_.adjoint();
        }

        auto desc_ptr = (std::complex<float>*)result.descriptor[e].data_ptr();
        for (int i = 0; i < unify_orb_num_; ++i) {
            for (int j = 0; j < unify_orb_num_; ++j) {
                desc_ptr[i * unify_orb_num_ + j] = std::complex<float>((float)transformed(i, j).real(), (float)transformed(i, j).imag());
            }
        }
        
        auto mask_ptr = (float*)result.mask[e].data_ptr();
        for (int i = 0; i < unify_orb_num_; ++i) {
            for (int j = 0; j < unify_orb_num_; ++j) {
                mask_ptr[i * unify_orb_num_ + j] = mask_f(i, j);
            }
        }

        result.edge_vec[e][0] = (float)edge.dist_vec(0);
        result.edge_vec[e][1] = (float)edge.dist_vec(1);
        result.edge_vec[e][2] = (float)edge.dist_vec(2);
        result.edge_src[e] = edge.atom_i;
        result.edge_dst[e] = edge.atom_j;
        result.ele_list[e] = {info_i.element, info_j.element};
    }
    
    return result;
}

