/**
 * @file neuronlayer.hpp
 * @brief Define NeuronLayer class
 * @author Dylan McGivern
 */
#pragma once

#include "neuron.hpp"

#include <vector>

/**
 * @class NeuronLayer
 * @brief Layer with 1 or more Neurons
 *
 * This class represents a layer of neurons in a network of neurons. The
 * layer can be given inputs, which will then return a number of outputs
 * that is equal to the amount of Neurons.
 */
class NeuronLayer {
  private:
    std::vector<Neuron> neurons;

  public:
    /**
     * @brief Constructs a layer of neurons
     *
     * @param neurons A list of neurons for this layer
     */
    NeuronLayer(const std::vector<Neuron> &neurons);
    /**
     * @brief Gives outputs given inputs
     *
     * @param inputs The inputs for this output
     * @return std::vector<double>: A list of outputs
     */
    std::vector<double> output(const std::vector<double> &inputs) const;
    /**
     * @brief Prints the layer in a readable manner
     */
    void __str__() const;
};
