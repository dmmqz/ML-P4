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

void Neuron::calcError(const std::vector<double> &inputs, const double &target) {
    double output = this->output(inputs);
    double sigmoid_deriv = output * (1 - output);

    this->error = sigmoid_deriv * -(target - output);
}

void Neuron::hiddenError(const std::vector<double> &inputs, const std::vector<double> &weights,
                         const std::vector<double> &jErrors) {
    double output = this->output(inputs);
    double sigmoid_deriv = output * (1 - output);

    double sum = 0;
    for (int i = 0; i < jErrors.size(); i++) {
        sum += jErrors[i] * weights[i];
    }
    this->error = sigmoid_deriv * sum;
}

double Neuron::gradient(const double iOutput) const { return (iOutput * this->error); }

std::vector<double> Neuron::delta(const std::vector<double> &inputs) const {
    // Includes bias as weights[0]
    std::vector<double> deltas(this->weights.size() + 1);
    double error = this->error;

    deltas[0] = this->learning_rate * error;

    for (int i = 0; i < deltas.size() - 1; i++) {
        deltas[i + 1] = this->learning_rate * inputs[i] * error;
    }

    return deltas;
}

void Neuron::storeNewWeights(const std::vector<double> &inputs) {
    std::vector<double> weights = this->delta(inputs);
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
