#include "include/neuronnetwork.hpp"
#include "include/neuronlayer.hpp"

#include <iostream>
#include <vector>

NeuronNetwork::NeuronNetwork(const std::vector<NeuronLayer> &layers) { this->layers = layers; }

std::vector<double> NeuronNetwork::feed_forward(std::vector<double> inputs) const {
    for (const NeuronLayer &layer : layers) {
        inputs = layer.output(inputs);
    }
    return inputs;
}

void NeuronNetwork::backpropagation(
    const std::pair<std::vector<double>, std::vector<double>> &trainingExample) {
    // Calculate new weights and bias for output layer
    NeuronLayer &outputLayer = this->layers[this->layers.size() - 1];

    // Get iOutputs
    std::vector<double> iOutputs(outputLayer.neurons.size());
    iOutputs = trainingExample.first;
    // Feed forward if there are hidden layers
    for (int i = 0; i < this->layers.size() - 1; i++) {
        iOutputs = this->layers[i].output(iOutputs);
    }

    // Store new weights and bias for output layer
    for (int i = 0; i < outputLayer.neurons.size(); i++) {
        outputLayer.neurons[i].calcError(iOutputs, trainingExample.second[i]);
        outputLayer.neurons[i].storeNewWeights(iOutputs, trainingExample.second[i], iOutputs[i]);
    }

    // Backpropagation for hidden layers
    for (int i = this->layers.size() - 2; i >= 0; i--) {
        NeuronLayer &layer = this->layers[i];
        NeuronLayer &nextLayer = this->layers[i + 1];

        std::vector<double> iOutputs = trainingExample.first;
        // Feed forward if there are previous hidden layers
        for (int j = 0; j < this->layers.size() - 3; j++) {
            iOutputs = this->layers[j].output(iOutputs);
        }

        // Store new weights and bias
        for (int j = 0; j < layer.neurons.size() - 1; j++) {
            // Get weights and errors from next layer
            std::vector<double> weights{};
            std::vector<double> jErrors{};
            for (Neuron &neuron : nextLayer.neurons) {
                weights.push_back(neuron.weights[j]);
                jErrors.push_back(neuron.error);
            }

            layer.neurons[j].hiddenError(iOutputs, weights, jErrors);
            layer.neurons[j].storeNewWeights(iOutputs, trainingExample.second[j], iOutputs[j]);
        }
    }

    // update weights and biases in whole network
    for (NeuronLayer &layer : this->layers) {
        for (Neuron &neuron : layer.neurons) {
            neuron.update();
        }
    }
}

void NeuronNetwork::__str__() const {
    for (auto &layer : this->layers) {
        std::cout << "Network:" << std::endl;
        layer.__str__();
    }
}
