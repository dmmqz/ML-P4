#include "src/include/neuron.hpp"
#include "src/include/neuronlayer.hpp"
#include "src/include/neuronnetwork.hpp"

#include <iostream>

int main() {
    Neuron andGate = Neuron({0.5, 0.5}, -1);
    std::cout << andGate.output({0, 0}) << std::endl;
    std::cout << andGate.output({0, 1}) << std::endl;
    std::cout << andGate.output({1, 0}) << std::endl;
    std::cout << andGate.output({1, 1}) << std::endl;

    return 0;
}
