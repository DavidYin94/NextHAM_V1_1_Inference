#pragma once

#include <string>
#include <vector>
#include <map>
#include <complex>

struct RVector {
    int x, y, z;
    bool operator<(const RVector& other) const {
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
};

struct CSRMatrix {
    std::vector<double> values;
    std::vector<std::complex<double>> complex_values;
    std::vector<int> indices;
    std::vector<int> indptr;
    int nnz;
};

struct CSRData {
    int nlocal;
    std::map<RVector, CSRMatrix> matrices;
    bool is_complex;
};

class CSRParser {
public:
    static CSRData parse(const std::string& filename, int nspin);
};
