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

void Neuron::__str__() const {
    std::cout << "Bias: " << this->bias << std::endl;
    std::cout << "Weights:" << std::endl;
    for (const double &weight : this->weights) {
        std::cout << weight << std::endl;
    }
}
