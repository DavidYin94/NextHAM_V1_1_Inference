#pragma once

#include <string>
#include <vector>
#include <Eigen/Dense>

struct Atom {
    std::string element;
    Eigen::Vector3d pos;
    // For move_x, move_y, move_z if needed
    int move[3];
};

struct Species {
    std::string element;
    double mass;
    std::string pseudo_pot;
    std::string orbital_file;
};

struct StruData {
    std::vector<Species> species;
    double lattice_constant;
    Eigen::Matrix3d lattice_vectors;
    std::vector<Atom> atoms;
    bool is_direct = false;
};

class StruParser {
public:
    static StruData parse(const std::string& filename);
};
