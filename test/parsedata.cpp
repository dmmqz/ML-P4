#include "include/parsedata.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::vector<double>> parseData(const std::string &filename) {
    // Get data file directory
    std::filesystem::path project_root = PROJ_ROOT;
    std::filesystem::path source_file = project_root / "data" / filename;

    std::ifstream file(source_file);
    std::vector<std::vector<double>> arr;

    if (file) {
        std::string line;

        std::string spl;
        char del = ',';

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<double> nums;

            while (std::getline(ss, spl, del)) {
                nums.push_back(std::stod(spl));
            }
            arr.push_back(nums);
        }
    }

    return arr;
}
