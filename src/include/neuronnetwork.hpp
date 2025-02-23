/**
 * @file neuronnetwork.hpp
 * @brief Define NeuronNetwork class
 * @author Dylan McGivern
 */
#pragma once

#include "neuronlayer.hpp"

#include <vector>

/**
 * @class NeuronNetwork
 * @brief Network with 1 or more NeuronLayers
 *
 * This class represents a network of layers with neurons. The network can
 * be given inputs, which will then return a number of outputs that is equal
 * to the amount of Neurons in the last layer.
 */
class NeuronNetwork {
  public:
    std::vector<NeuronLayer> layers;
    /**
     * @brief Constructs a network of neurons
     *
     * @param neurons A list of neurons for this layer
     */
    NeuronNetwork(const std::vector<NeuronLayer> &layers);
    /**
     * @brief Gives outputs given inputs
     *
     * @param inputs The inputs for this output
     * @return std::vector<double>: A list of outputs
     */
    std::vector<double> feed_forward(std::vector<double> inputs) const;
    void backpropagation(const std::vector<double> &input, const std::vector<double> &target);
    void train(const std::vector<std::vector<double>> &inputs,
               const std::vector<std::vector<double>> &targets);
    /**
     * @brief Prints the network in a readable manner
     */
    void __str__() const;
};
