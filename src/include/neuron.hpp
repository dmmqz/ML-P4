#pragma once

#include <vector>

class Neuron {
  private:
    std::vector<double> weights;
    double bias;

  public:
    Neuron(const std::vector<double> &weights, const double &bias);
    bool output(const std::vector<bool> &inputs) const;
    void __str__() const;
};
