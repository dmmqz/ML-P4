#include "include/neuron.hpp"

#include <cmath>
#include <iostream>
#include <vector>

Neuron::Neuron(const std::vector<double> &weights, const double &bias) {
    this->weights = weights;
    this->bias = bias;
    this->newWeights.resize(this->weights.size() + 1);
}

double Neuron::output(const std::vector<double> &inputs) const {
    double dot_product = 0;
    for (int i = 0; i < this->weights.size(); i++) {
        dot_product += inputs[i] * this->weights[i];
    }

    // sigmoid function
    double sigmoid = 1 / (1 + std::exp(-(dot_product + this->bias)));

    return sigmoid;
}

double Neuron::calcError(const std::vector<double> &inputs, const double &target) {
    double output = this->output(inputs);
    double sigmoid_deriv = output * (1 - output);

    this->error = sigmoid_deriv * -(target - output);
    return this->error;
}

double Neuron::hiddenError(const std::vector<double> &inputs, const std::vector<double> &weights,
                           const std::vector<double> &jErrors) {
    double output = this->output(inputs);
    double sigmoid_deriv = output * (1 - output);

    double sum = 0;
    for (int i = 0; i < jErrors.size(); i++) {
        sum += jErrors[i] * weights[i];
    }
    this->error = sigmoid_deriv * sum;
    return this->error;
}

double Neuron::gradient(const std::vector<double> &inputs, const double &target,
                        const double iOutput) const {
    return (iOutput * this->error);
}

std::vector<double> Neuron::delta(const std::vector<double> &inputs, const double &target,
                                  const std::vector<double> iOutputs) const {
    // Includes bias as weights[0]
    std::vector<double> weights(this->weights.size() + 1);
    double error = this->error;

    weights[0] = this->learning_rate * error;

    for (int i = 0; i < weights.size() - 1; i++) {
        weights[i + 1] = this->learning_rate * iOutputs[i] * error;
    }

    return weights;
}

void Neuron::storeNewWeights(const std::vector<double> &inputs, const double &target,
                             const std::vector<double> iOutputs) {
    std::vector<double> weights = this->delta(inputs, target, iOutputs);
    this->newWeights[0] = this->bias - weights[0];

    for (int i = 0; i < this->weights.size(); i++) {
        this->newWeights[i + 1] = this->weights[i] - weights[i + 1];
    }
}

void Neuron::update() {
    this->bias = this->newWeights[0];

    for (int i = 0; i < this->weights.size(); i++) {
        this->weights[i] = this->newWeights[i + 1];
    }
}

void Neuron::__str__() const {
    std::cout << "Bias: " << this->bias << std::endl;
    std::cout << "Weights:" << std::endl;
    for (const double &weight : this->weights) {
        std::cout << weight << std::endl;
    }
}
