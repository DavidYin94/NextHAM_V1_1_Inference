#ifndef BLOCK_PARSER_H
#define BLOCK_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <complex>
#include <filesystem>
#include "csr_parser.h" // For RVector and other shared types if needed

namespace fs = std::filesystem;

struct BlockData {
    RVector r;
    std::vector<int> row_idx;
    std::vector<int> col_idx;
    std::vector<double> values;
    std::vector<std::complex<double>> complex_values;
};

struct BlockParserResult {
    // r_vec -> (pair_ai_aj -> list of blocks)
    std::map<RVector, std::map<std::pair<int, int>, std::vector<BlockData>>> matrices;
    bool is_complex;
};

class BlockParser {
public:
    static BlockParserResult parse_dir(const std::string& dir_path, const std::string& prefix, int nspin);
private:
    static std::pair<std::map<std::pair<int, int>, std::vector<BlockData>>, int> 
    parse_file(const std::string& filename, bool is_binary, int nspin);
};

#endif
