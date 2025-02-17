#pragma once

#include "neuron.hpp"

#include <vector>

class NeuronLayer {
  private:
    std::vector<Neuron> neurons;

  public:
    NeuronLayer(const std::vector<Neuron> &neurons);
    std::vector<bool> output(const std::vector<bool> &inputs) const;
    void __str__() const;
};
