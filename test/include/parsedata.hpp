/**
 * @file parsedata.hpp
 * @brief Helper functions for data parsing
 * @author Dylan McGivern
 */
#include <string>
#include <vector>

/**
 * @brief Converts data from a CSV to a vector
 *
 * @param filename The name of the data file
 * @return std::vector<std::vector<double>>: The data transformed into a vector
 */
std::vector<std::vector<double>> parseData(const std::string &filename);
