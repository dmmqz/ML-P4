#include "neuronlayer.hpp"

#include <iostream>
#include <vector>

NeuronLayer::NeuronLayer(const std::vector<Neuron> &neurons) {
    this->neurons = neurons;
}

std::vector<bool> NeuronLayer::output(const std::vector<bool> &inputs) const {
    std::vector<bool> outputs{};
    for (const Neuron &neuron : this->neurons) {
        outputs.push_back(neuron.output(inputs));
    }
    return outputs;
}

void NeuronLayer::__str__() const {
    for (const Neuron &neuron : this->neurons) {
        std::cout << "Neuron:" << std::endl;
        neuron.__str__();
    }
}
