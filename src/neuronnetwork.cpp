#include "neuronnetwork.hpp"

#include <iostream>
#include <vector>

NeuronNetwork::NeuronNetwork(const std::vector<NeuronLayer> layers) {
    this->layers = layers;
}

std::vector<bool> NeuronNetwork::feed_forward(std::vector<bool> inputs) const {
    for (const NeuronLayer &layer : layers) {
        inputs = layer.output(inputs);
    }
    return inputs;
}

void NeuronNetwork::__str__() const {
    for (auto &layer : this->layers) {
        std::cout << "Network:" << std::endl;
        layer.__str__();
    }
}
