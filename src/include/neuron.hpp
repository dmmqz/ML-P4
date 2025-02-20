/**
 * @file neuron.hpp
 * @brief Define Neuron class
 * @author Dylan McGivern
 */
#pragma once

#include <vector>

/**
 * @class Neuron
 * @brief Neuron with 1 or more inputs
 *
 * This class represents a neuron in a layer. This class takes weights and
 * bias as input. Once the class is initialized, output can be called. The
 * amount of inputs must be as much or less than the amount of weights.
 */
class Neuron {
  private:
    std::vector<double> weights;
    double bias;

  public:
    /**
     * @brief Constructs a Neuron
     *
     * @param weights A list of all the weights
     * @param bias The bias for the Neuron
     */
    Neuron(const std::vector<double> &weights, const double &bias);
    /**
     * @brief Gives an output given an input
     *
     * @param inputs Inputs to check with
     *
     * @return bool: The truth value given the input
     */
    bool output(const std::vector<bool> &inputs) const;
    /**
     * @brief Prints the Neuron in a readable manner
     */
    void __str__() const;
};
