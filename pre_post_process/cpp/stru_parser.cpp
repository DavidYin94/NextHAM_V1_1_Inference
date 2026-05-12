#include "stru_parser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

StruData StruParser::parse(const std::string& filename) {
    StruData data;
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(f, line)) {
        // Simple trimmer
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (line.empty() || line[0] == '#') continue;

        if (line == "ATOMIC_SPECIES") {
            while (std::getline(f, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty() || line[0] == '#') continue;
                if (line == "NUMERICAL_ORBITAL" || line == "LATTICE_CONSTANT" || line == "LATTICE_VECTORS" || line == "ATOMIC_POSITIONS") break;

                std::stringstream ss(line);
                Species s;
                if (ss >> s.element >> s.mass >> s.pseudo_pot) {
                    data.species.push_back(s);
                }
                if (data.species.size() > 0 && (line.find("NUMERICAL_ORBITAL") != std::string::npos)) break;
            }
        }
        
        if (line == "NUMERICAL_ORBITAL") {
            int i = 0;
            while (std::getline(f, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty() || line[0] == '#') continue;
                if (line == "LATTICE_CONSTANT" || line == "LATTICE_VECTORS" || line == "ATOMIC_POSITIONS") break;
                if (i < data.species.size()) {
                    data.species[i].orbital_file = line;
                    i++;
                }
            }
        }

        if (line == "LATTICE_CONSTANT") {
            while (std::getline(f, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty() || line[0] == '#') continue;
                data.lattice_constant = std::stod(line.substr(0, line.find_first_of(" \t#")));
                // Convert Bohr to Angstrom (match ASE)
                data.lattice_constant *= 0.529177210903;
                break;
            }
        }

        if (line == "LATTICE_VECTORS") {
            for (int i = 0; i < 3; ++i) {
                while (std::getline(f, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty() || line[0] == '#') continue;
                    std::stringstream ss(line);
                    ss >> data.lattice_vectors(i, 0) >> data.lattice_vectors(i, 1) >> data.lattice_vectors(i, 2);
                    break;
                }
            }
        }

        if (line == "ATOMIC_POSITIONS") {
            std::string coord_type;
            while (std::getline(f, line)) {
                line.erase(0, line.find_first_not_of(" \t\r\n"));
                line.erase(line.find_last_not_of(" \t\r\n") + 1);
                if (line.empty() || line[0] == '#') continue;
                coord_type = line;
                break;
            }
            if (coord_type == "Direct") data.is_direct = true;
            
            for (auto& s : data.species) {
                // Find species name header
                while (std::getline(f, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty() || line[0] == '#') continue;
                    std::stringstream ss(line);
                    std::string species_name;
                    ss >> species_name;
                    if (species_name == s.element) break;
                }
                // Magnetic moment
                while (std::getline(f, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty() || line[0] == '#') continue;
                    break; 
                }
                // Number of atoms
                int num_atoms = 0;
                while (std::getline(f, line)) {
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
                    if (line.empty() || line[0] == '#') continue;
                    num_atoms = std::stoi(line);
                    break;
                }
                for (int i = 0; i < num_atoms; ++i) {
                    while (std::getline(f, line)) {
                        line.erase(0, line.find_first_not_of(" \t\r\n"));
                        line.erase(line.find_last_not_of(" \t\r\n") + 1);
                        if (line.empty() || line[0] == '#') continue;
                        Atom a;
                        a.element = s.element;
                        std::stringstream ss(line);
                        ss >> a.pos(0) >> a.pos(1) >> a.pos(2);
                        data.atoms.push_back(a);
                        break;
                    }
                }
            }
        }
    }
    return data;
}
