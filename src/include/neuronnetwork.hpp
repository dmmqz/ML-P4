#pragma once

#include "neuronlayer.hpp"

#include <vector>

class NeuronNetwork {
  private:
    std::vector<NeuronLayer> layers;

  public:
    NeuronNetwork(const std::vector<NeuronLayer> layers);
    std::vector<bool> feed_forward(std::vector<bool> inputs) const;
    void __str__() const;
};
