#include "include/neuron.hpp"

#include <cmath>
#include <iostream>
#include <vector>

Neuron::Neuron(const std::vector<double> &weights, const double &bias) {
    this->weights = weights;
    this->bias = bias;
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

double Neuron::error(const std::vector<double> &inputs, const double &target) const {
    double output = this->output(inputs);

    double sigmoid_deriv = output * (1 - output);
    return (sigmoid_deriv * -(target - output));
}

double Neuron::gradient(const std::vector<double> &inputs, const double &target,
                        const double iOutput) const {
    return (iOutput * this->error(inputs, target));
}

std::vector<double> Neuron::delta(const std::vector<double> &inputs, const double &target,
                                  const double iOutput) const {
    // Includes bias as weights[0]
    std::vector<double> weights(this->weights.size() + 1);
    double error = this->error(inputs, target);

    weights[0] = this->learning_rate * error;

    for (int i = 1; i < this->weights.size(); i++) {
        weights[i] = this->learning_rate * iOutput * error;
    }

    return weights;
}

void Neuron::update(const std::vector<double> &inputs, const double &target, const double iOutput) {
    std::vector<double> weights = this->delta(inputs, target, iOutput);
    this->bias -= weights[0];

    for (int i = 1; i < this->weights.size(); i++) {
        this->weights[i - 1] -= weights[i];
    }
}

void Neuron::__str__() const {
    std::cout << "Bias: " << this->bias << std::endl;
    std::cout << "Weights:" << std::endl;
    for (const double &weight : this->weights) {
        std::cout << weight << std::endl;
    }
}
