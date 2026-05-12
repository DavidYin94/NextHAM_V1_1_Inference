#include "csr_parser.h"
#include <fstream>
#include <iostream>
#include <sstream>

CSRData CSRParser::parse(const std::string& filename, int nspin) {
    CSRData data;
    data.is_complex = (nspin == 4);
    
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    char header[4];
    f.read(header, 4);
    bool detected_binary = true;
    if (std::string(header, 4) == "STEP") {
        detected_binary = false;
    }
    f.seekg(0);

    if (detected_binary) {
        int step, nlocal, nR;
        f.read((char*)&step, 4);
        f.read((char*)&nlocal, 4);
        f.read((char*)&nR, 4);
        data.nlocal = nlocal;

        for (int i = 0; i < nR; ++i) {
            RVector r;
            int nnz;
            f.read((char*)&r.x, 4);
            f.read((char*)&r.y, 4);
            f.read((char*)&r.z, 4);
            f.read((char*)&nnz, 4);

            CSRMatrix m;
            m.nnz = nnz;
            if (nnz > 0) {
                if (data.is_complex) {
                    m.complex_values.resize(nnz);
                    f.read((char*)m.complex_values.data(), nnz * 16);
                } else {
                    m.values.resize(nnz);
                    f.read((char*)m.values.data(), nnz * 8);
                }
                m.indices.resize(nnz);
                f.read((char*)m.indices.data(), nnz * 4);
                m.indptr.resize(nlocal + 1);
                f.read((char*)m.indptr.data(), (nlocal + 1) * 4);
            } else {
                // IMPORTANT: When nnz is 0, ABACUS binary format STILL writes 
                // the indptr block (size nlocal+1) containing all zeros or similar.
                // We MUST skip/read it to keep the file pointer aligned for the next R.
                std::vector<int> dummy_indptr(nlocal + 1);
                f.read((char*)dummy_indptr.data(), (nlocal + 1) * 4);
                m.indptr = std::move(dummy_indptr);
            }
            data.matrices[r] = std::move(m);
        }
    } else {
        std::ifstream tf(filename);
        std::string line;
        int nlocal = 0;
        int nR = 0;
        while (std::getline(tf, line)) {
            if (line.find("dimension") != std::string::npos) {
                nlocal = std::stoi(line.substr(line.find_last_of(" \t") + 1));
            }
            if (line.find("number") != std::string::npos) {
                nR = std::stoi(line.substr(line.find_last_of(" \t") + 1));
                break;
            }
        }
        data.nlocal = nlocal;

        for (int i = 0; i < nR; ++i) {
            RVector r;
            int nnz;
            while (std::getline(tf, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::stringstream ss(line);
                if (ss >> r.x >> r.y >> r.z >> nnz) break;
            }

            CSRMatrix m;
            m.nnz = nnz;
            if (nnz > 0) {
                std::getline(tf, line);
                std::stringstream ss_vals(line);
                if (data.is_complex) {
                    m.complex_values.reserve(nnz);
                    std::string val_str;
                    while (ss_vals >> val_str) {
                        // format (real,imag)
                        size_t comma = val_str.find(',');
                        double real = std::stod(val_str.substr(1, comma - 1));
                        double imag = std::stod(val_str.substr(comma + 1, val_str.size() - comma - 2));
                        m.complex_values.emplace_back(real, imag);
                    }
                } else {
                    m.values.reserve(nnz);
                    double val;
                    while (ss_vals >> val) m.values.push_back(val);
                }

                std::getline(tf, line);
                std::stringstream ss_idx(line);
                m.indices.reserve(nnz);
                int max_idx = -1;
                int idx;
                while (ss_idx >> idx) {
                    m.indices.push_back(idx);
                    if (idx > max_idx) max_idx = idx;
                }
                // std::cout << "R=(" << r.x << "," << r.y << "," << r.z << ") nnz=" << nnz << " max_idx=" << max_idx << std::endl;

                std::getline(tf, line);
                std::stringstream ss_ptr(line);
                m.indptr.reserve(nlocal + 1);
                int ptr;
                while (ss_ptr >> ptr) m.indptr.push_back(ptr);
            }
            data.matrices[r] = std::move(m);
        }
    }
    return data;
}
