#pragma once

#include "stru_parser.h"
#include <vector>
#include <Eigen/Dense>

struct Edge {
    int atom_i;
    int atom_j;
    Eigen::Vector3d dist_vec;
    Eigen::Vector3i r_offset;
    double dist;
};

class NeighborList {
public:
    static std::vector<Edge> build(const StruData& data, double cutoff);
};
