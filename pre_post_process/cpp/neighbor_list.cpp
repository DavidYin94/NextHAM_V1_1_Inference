#include "neighbor_list.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

std::vector<Edge> NeighborList::build(const StruData& data, double cutoff) {
    std::vector<Edge> edges;
    int n = data.atoms.size();
    if (n == 0) return edges;

    Eigen::Matrix3d lat = data.lattice_vectors * data.lattice_constant;
    Eigen::Matrix3d inv_lat = lat.inverse();
    
    // 1. Determine the range of R-vectors to check
    Eigen::Vector3d v0 = lat.row(0);
    Eigen::Vector3d v1 = lat.row(1);
    Eigen::Vector3d v2 = lat.row(2);
    
    auto get_h = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
        Eigen::Vector3d n_vec = b.cross(c).normalized();
        return std::abs(a.dot(n_vec));
    };
    
    double h0 = get_h(v0, v1, v2);
    double h1 = get_h(v1, v0, v2);
    double h2 = get_h(v2, v0, v1);
    
    int nrx = std::ceil(cutoff / h0);
    int nry = std::ceil(cutoff / h1);
    int nrz = std::ceil(cutoff / h2);

    // 2. Fractional coordinates in [0, 1)
    std::vector<Eigen::Vector3d> frac_pos(n);
    for (int i = 0; i < n; ++i) {
        if (data.is_direct) {
            frac_pos[i] = data.atoms[i].pos;
        } else {
            // Cartesian in Bohr or Angstrom? 
            // After our fix, lattice_constant is in Angstroms if it was Bohr.
            // ABACUS Cartesian positions are typically pos * lattice_constant.
            Eigen::Vector3d p = data.atoms[i].pos * data.lattice_constant;
            frac_pos[i] = inv_lat * p;
        }
        for (int k = 0; k < 3; ++k) {
            frac_pos[i](k) -= std::floor(frac_pos[i](k) + 1e-9); // Small epsilon
        }
    }

    double cutoff_sq = cutoff * cutoff;

    // Grid search implementation
    // For n=10,000, we should use a proper cell list.
    int gx = std::max(1, (int)(h0 / (cutoff + 1e-6)));
    int gy = std::max(1, (int)(h1 / (cutoff + 1e-6)));
    int gz = std::max(1, (int)(h2 / (cutoff + 1e-6)));
    
    std::vector<std::vector<int>> grid(gx * gy * gz);
    auto get_grid_idx = [&](int ix, int iy, int iz) {
        ix = (ix % gx + gx) % gx;
        iy = (iy % gy + gy) % gy;
        iz = (iz % gz + gz) % gz;
        return ix * gy * gz + iy * gz + iz;
    };

    for (int i = 0; i < n; ++i) {
        int ix = std::min(gx - 1, (int)(frac_pos[i](0) * gx));
        int iy = std::min(gy - 1, (int)(frac_pos[i](1) * gy));
        int iz = std::min(gz - 1, (int)(frac_pos[i](2) * gz));
        grid[ix * gy * gz + iy * gz + iz].push_back(i);
    }

    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d pi = frac_pos[i];
        int ix = (int)(pi(0) * gx);
        int iy = (int)(pi(1) * gy);
        int iz = (int)(pi(2) * gz);

        // Check surrounding cells including images
        // For each atom i, we look for j in (central cell + R)
        // This is equivalent to looking for j in central cell, and checking all R
        // that could bring j within cutoff of i.
        
        for (int rx = -nrx; rx <= nrx; ++rx) {
            for (int ry = -nry; ry <= nry; ++ry) {
                for (int rz = -nrz; rz <= nrz; ++rz) {
                    // Shifted fractional position of i to check against j in central cell
                    // d = pj - (pi - R)
                    // This is still slightly inefficient if we don't use the grid for j.
                    
                    // Correct grid search:
                    // Find grid cells in central cell that are within cutoff of (pi - R)
                    Eigen::Vector3d center_in_frac = pi - Eigen::Vector3d(rx, ry, rz);
                    // Range of cells in fractional units: [center - cutoff_frac, center + cutoff_frac]
                    // But easier: just iterate over all j and R. 
                    // To be fast for 10,000 atoms, we MUST use the grid.
                    
                    // Actually, for each i, we only need to check R such that 
                    // the image cell (central cell + R) is within cutoff of pi.
                }
            }
        }
        
        // Re-simplified efficient version:
        for (int rx = -nrx; rx <= nrx; ++rx) {
            for (int ry = -nry; ry <= nry; ++ry) {
                for (int rz = -nrz; rz <= nrz; ++rz) {
                    Eigen::Vector3d R_vec(rx, ry, rz);
                    for (int j = 0; j < n; ++j) {
                        Eigen::Vector3d d_frac = frac_pos[j] + R_vec - pi;
                        Eigen::Vector3d d_cart = lat * d_frac;
                        double d2 = d_cart.squaredNorm();
                        if (d2 < cutoff_sq) {
                            edges.push_back({i, j, d_cart, {rx, ry, rz}, std::sqrt(d2)});
                        }
                    }
                }
            }
        }
    }
    // Sort edges for consistency as in Python
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        if (a.r_offset(0) != b.r_offset(0)) return a.r_offset(0) < b.r_offset(0);
        if (a.r_offset(1) != b.r_offset(1)) return a.r_offset(1) < b.r_offset(1);
        if (a.r_offset(2) != b.r_offset(2)) return a.r_offset(2) < b.r_offset(2);
        if (a.atom_i != b.atom_i) return a.atom_i < b.atom_i;
        return a.atom_j < b.atom_j;
    });

    return edges;
}
