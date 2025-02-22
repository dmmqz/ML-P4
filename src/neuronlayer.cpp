#include "include/neuronlayer.hpp"
#include "include/neuron.hpp"

#include <iostream>
#include <vector>

NeuronLayer::NeuronLayer(const std::vector<Neuron> &neurons) {
    this->neurons = neurons;
}

std::vector<double> NeuronLayer::output(const std::vector<double> &inputs) const {
    std::vector<double> outputs{};
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
