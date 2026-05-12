#include "stru_parser.h"
#include "csr_parser.h"
#include "block_parser.h"
#include "neighbor_list.h"
#include "transformer.h"
#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <filesystem>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <stru_file> <data_dir> <nspin> <cutoff> <output_pth>" << std::endl;
        return 1;
    }

    std::string stru_file = argv[1];
    std::string data_dir = argv[2];
    int nspin = std::stoi(argv[3]);
    double cutoff = std::stod(argv[4]);
    std::string output_pth = argv[5];

    try {
        auto total_start = std::chrono::high_resolution_clock::now();

        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "Parsing STRU..." << std::endl;
        auto stru_data = StruParser::parse(stru_file);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "  STRU parsing took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

        TransformationEngine::ProcessedData processed;
        TransformationEngine engine(nspin);

        std::string csr_file = data_dir + "/hrs1_nao.csr";
        if (fs::exists(csr_file)) {
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Parsing CSR..." << std::endl;
            auto csr_data = CSRParser::parse(csr_file, nspin);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  CSR parsing took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Building Neighbor List..." << std::endl;
            auto edges = NeighborList::build(stru_data, cutoff);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Neighbor list build took: " << std::chrono::duration<double>(end - start).count() << " s (" << edges.size() << " edges found)" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Transforming data..." << std::endl;
            processed = engine.process(stru_data, csr_data, edges);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Transformation took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;
        } else {
            start = std::chrono::high_resolution_clock::now();
            std::cout << "Parsing Blocks (prefix hrs_block_up)..." << std::endl;
            auto block_data = BlockParser::parse_dir(data_dir, "hrs_block_up", nspin);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Block parsing took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Building Neighbor List..." << std::endl;
            auto edges = NeighborList::build(stru_data, cutoff);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Neighbor list build took: " << std::chrono::duration<double>(end - start).count() << " s (" << edges.size() << " edges found)" << std::endl;

            start = std::chrono::high_resolution_clock::now();
            std::cout << "Transforming data..." << std::endl;
            processed = engine.process(stru_data, block_data, edges);
            end = std::chrono::high_resolution_clock::now();
            std::cout << "  Transformation took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;
        }

        start = std::chrono::high_resolution_clock::now();
        std::cout << "Saving to " << output_pth << "..." << std::endl;
        
        // Match Python's [descriptor, overlap, mask, edge_vec, edge_src, edge_dst, ele_list, output_path_str]
        c10::impl::GenericList list(c10::AnyType::get());
        list.push_back(processed.descriptor);
        list.push_back(torch::IValue()); // Set overlap to None
        list.push_back(processed.mask);
        list.push_back(processed.edge_vec);
        list.push_back(processed.edge_src);
        list.push_back(processed.edge_dst);
        
        // ele_list (list of lists of strings, to be interpreted as list of tuples in Python)
        c10::impl::GenericList ele_ivalue_list(c10::AnyType::get());
        for (const auto& p : processed.ele_list) {
            c10::impl::GenericList tuple_list(c10::AnyType::get());
            tuple_list.push_back(p.first);
            tuple_list.push_back(p.second);
            ele_ivalue_list.push_back(tuple_list);
        }
        list.push_back(ele_ivalue_list);
        list.push_back(std::string("output_inference.pth"));

        // Match Python's (data_list, label) tuple
        std::vector<torch::IValue> tuple_elements;
        tuple_elements.push_back(list);
        tuple_elements.push_back(torch::IValue()); // label is None
        auto root_tuple = c10::ivalue::Tuple::create(std::move(tuple_elements));

        // Use pickle_save to match Python's torch.save
        auto bytes = torch::jit::pickle_save(torch::IValue(root_tuple));
        std::ofstream fout(output_pth, std::ios::binary);
        fout.write(bytes.data(), bytes.size());
        fout.close();
        end = std::chrono::high_resolution_clock::now();
        std::cout << "  Saving took: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;

        auto total_end = std::chrono::high_resolution_clock::now();
        std::cout << "Total Preprocessing time: " << std::chrono::duration<double>(total_end - total_start).count() << " s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

