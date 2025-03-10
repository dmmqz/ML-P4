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
  public:
    double learning_rate = 1;
    std::vector<double> weights;
    std::vector<double> newWeights;
    double bias;
    double error;
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
     * @return double: The truth value given the input
     */
    double output(const std::vector<double> &inputs) const;
    /**
     * @brief Calculates error for an output Neuron
     *
     * @param inputs The inputs for this Neuron
     * @param target The target for this Neuron
     */
    void calcError(const std::vector<double> &inputs, const double &target);
    /**
     * @brief Calculates error for a hidden Neuron
     *
     * @param inputs The inputs for this Neuron
     * @param target The target for this Neuron
     * @param jErrors Errors of the next Neuron
     */
    void hiddenError(const std::vector<double> &inputs, const std::vector<double> &weights,
                     const std::vector<double> &jErrors);
    /**
     * @brief Calculates gradient to adjust this Neuron with
     *
     * @param iOutput Output from previous Neuron
     *
     * @return double: Gradient used in `delta` method
     */
    double gradient(const double iOutput) const;
    /**
     * @brief Calculates change in weights for Neuron
     *
     * @param inputs The inputs to check for
     *
     * @return std::vector<double>: Deltas to calculate new weights with
     */
    std::vector<double> delta(const std::vector<double> &inputs) const;
    /**
     * @brief Calculates and stores new weights
     *
     * @param inputs The inputs to check for
     */
    void storeNewWeights(const std::vector<double> &inputs);
    /**
     * @brief Changes `weights` attribute to `newWeights`
     */
    void update();
    /**
     * @brief Prints the Neuron in a readable manner
     */
    void __str__() const;
};
