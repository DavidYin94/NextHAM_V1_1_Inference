#pragma once

#include "stru_parser.h"
#include "csr_parser.h"
#include "neighbor_list.h"
#include "block_parser.h"
#include <torch/torch.h>
#include <map>
#include <string>

class TransformationEngine {
public:
    TransformationEngine(int nspin);
    
    struct ProcessedData {
        torch::Tensor descriptor;
        torch::Tensor overlap;
        torch::Tensor mask;
        torch::Tensor edge_vec;
        torch::Tensor edge_src;
        torch::Tensor edge_dst;
        std::vector<std::pair<std::string, std::string>> ele_list;
    };

    ProcessedData process(const StruData& stru, const CSRData& csr, const std::vector<Edge>& edges);
    ProcessedData process(const StruData& stru, const BlockParserResult& blocks, const std::vector<Edge>& edges);

private:
    int nspin_;
    int unify_orb_num_;
    Eigen::MatrixXcd trans_mat_;
    
    std::map<std::string, int> orb_origin_;
    std::vector<int> get_orbital_patch_pattern(int orb_type);
    Eigen::MatrixXcd get_transform_matrix();
    
    struct AtomInfo {
        std::string element;
        std::vector<int> orbital_patch;
        std::vector<int> orbital_index;
    };
};
