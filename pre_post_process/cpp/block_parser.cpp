#include "block_parser.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <glob.h>
#include <algorithm>

BlockParserResult BlockParser::parse_dir(const std::string& dir_path, const std::string& prefix, int nspin) {
    BlockParserResult result;
    result.is_complex = (nspin == 4);

    std::string pattern = dir_path + "/" + prefix + "_*.dat";
    glob_t glob_result;
    if (glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result) != 0) {
        globfree(&glob_result);
        return result;
    }

    if (glob_result.gl_pathc == 0) {
        globfree(&glob_result);
        return result;
    }

    // Auto-detect binary
    bool is_binary = false;
    {
        std::ifstream f(glob_result.gl_pathv[0], std::ios::binary);
        if (f.is_open()) {
            char header[4];
            f.read(header, 4);
            if (std::string(header, 4) != "STEP") {
                is_binary = true;
            }
        }
    }

    double threshold = 1e-10;

    for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
        auto [file_data, step] = parse_file(glob_result.gl_pathv[i], is_binary, nspin);
        for (auto const& [pair, blocks] : file_data) {
            for (auto const& block : blocks) {
                // Apply threshold (same as Python)
                bool has_significant = false;
                if (result.is_complex) {
                    for (const auto& v : block.complex_values) {
                        if (std::abs(v) > threshold) {
                            has_significant = true;
                            break;
                        }
                    }
                } else {
                    for (const auto& v : block.values) {
                        if (std::abs(v) > threshold) {
                            has_significant = true;
                            break;
                        }
                    }
                }

                if (has_significant) {
                    result.matrices[block.r][pair].push_back(block);
                }
            }
        }
    }
    
    globfree(&glob_result);
    return result;
}

std::pair<std::map<std::pair<int, int>, std::vector<BlockData>>, int> 
BlockParser::parse_file(const std::string& filename, bool is_binary, int nspin) {
    std::map<std::pair<int, int>, std::vector<BlockData>> file_data;
    int step = 0;

    if (is_binary) {
        std::ifstream f(filename, std::ios::binary);
        if (!f.is_open()) return {file_data, 0};

        int n_ap;
        f.read((char*)&step, 4);
        f.read((char*)&n_ap, 4);

        for (int i = 0; i < n_ap; ++i) {
            int ai, aj, rs, cs, nr;
            f.read((char*)&ai, 4);
            f.read((char*)&aj, 4);
            f.read((char*)&rs, 4);
            f.read((char*)&cs, 4);
            f.read((char*)&nr, 4);
            std::pair<int, int> pair_key = {ai, aj};

            std::vector<int> row_idx(rs);
            f.read((char*)row_idx.data(), rs * 4);
            std::vector<int> col_idx(cs);
            f.read((char*)col_idx.data(), cs * 4);

            for (int j = 0; j < nr; ++j) {
                BlockData block;
                f.read((char*)&block.r.x, 4);
                f.read((char*)&block.r.y, 4);
                f.read((char*)&block.r.z, 4);
                block.row_idx = row_idx;
                block.col_idx = col_idx;

                int block_size = rs * cs;
                if (nspin == 4) {
                    block.complex_values.resize(block_size);
                    f.read((char*)block.complex_values.data(), block_size * 16);
                } else {
                    block.values.resize(block_size);
                    f.read((char*)block.values.data(), block_size * 8);
                }
                file_data[pair_key].push_back(std::move(block));
            }
        }
    } else {
        std::ifstream f(filename);
        if (!f.is_open()) return {file_data, 0};

        std::string line;
        if (!std::getline(f, line)) return {file_data, 0};
        std::stringstream ss_step(line);
        std::string dummy;
        ss_step >> dummy >> step;

        if (!std::getline(f, line)) return {file_data, 0};
        std::stringstream ss_nap(line);
        int n_ap;
        ss_nap >> dummy >> n_ap;

        for (int i = 0; i < n_ap; ++i) {
            if (!std::getline(f, line)) break;
            std::stringstream ss_pair(line);
            int ai, aj, rs, cs, nr;
            ss_pair >> dummy >> ai >> aj >> rs >> cs >> nr;
            std::pair<int, int> pair_key = {ai, aj};

            std::getline(f, line); // row_idx line
            std::stringstream ss_row(line);
            ss_row >> dummy;
            std::vector<int> row_idx;
            int idx;
            while (ss_row >> idx) row_idx.push_back(idx);

            std::getline(f, line); // col_idx line
            std::stringstream ss_col(line);
            ss_col >> dummy;
            std::vector<int> col_idx;
            while (ss_col >> idx) col_idx.push_back(idx);

            for (int j = 0; j < nr; ++j) {
                if (!std::getline(f, line)) break;
                std::stringstream ss_R(line);
                BlockData block;
                ss_R >> dummy >> block.r.x >> block.r.y >> block.r.z;
                block.row_idx = row_idx;
                block.col_idx = col_idx;

                int block_size = rs * cs;
                if (nspin == 4) {
                    block.complex_values.reserve(block_size);
                    while (block.complex_values.size() < (size_t)block_size) {
                        if (!std::getline(f, line)) break;
                        std::stringstream ss_data(line);
                        double real, imag;
                        char comma, p1, p2;
                        while (ss_data >> p1 >> real >> comma >> imag >> p2) {
                            block.complex_values.emplace_back(real, imag);
                        }
                    }
                } else {
                    block.values.reserve(block_size);
                    while (block.values.size() < (size_t)block_size) {
                        if (!std::getline(f, line)) break;
                        std::stringstream ss_data(line);
                        double val;
                        while (ss_data >> val) block.values.push_back(val);
                    }
                }
                file_data[pair_key].push_back(std::move(block));
            }
        }
    }

    return {file_data, step};
}
