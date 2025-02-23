#define CATCH_CONFIG_MAIN

#include "include/catch.hpp"

#include "../src/include/neuron.hpp"
#include "../src/include/neuronlayer.hpp"
#include "../src/include/neuronnetwork.hpp"

#include <cmath>

TEST_CASE("Backpropagation AND-gate (single iteration)") {
    NeuronNetwork andGate = NeuronNetwork({NeuronLayer({Neuron({-0.5, 0.5}, 1.5)})});

    std::vector<std::vector<double>> inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<double> targets = {0, 0, 0, 1};
    std::vector<double> expectedOutputs = {0.818, 0.731, 0.881, 0.818};

    // Check feed forward (normal sigmoid function)
    CHECK(std::abs(andGate.feed_forward(inputs[0])[0] - expectedOutputs[0]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[1])[0] - expectedOutputs[1]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[2])[0] - expectedOutputs[2]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[3])[0] - expectedOutputs[3]) < 0.01);

    // Test backpropagation
    std::pair<std::vector<double>, std::vector<double>> trainingExample;
    std::vector<double> target = {targets[0]};
    trainingExample.first = inputs[0];
    trainingExample.second = target;

    andGate.backpropagation(trainingExample);

    std::vector<double> expectedWeights = {1.378, -0.5, 0.5};

    // Test if biases and weights are expected weights
    CHECK(std::abs(andGate.layers[0].neurons[0].bias - expectedWeights[0]) < 0.01);
    CHECK(std::abs(andGate.layers[0].neurons[0].weights[0] - expectedWeights[1]) < 0.01);
    CHECK(std::abs(andGate.layers[0].neurons[0].weights[1] - expectedWeights[2]) < 0.01);
}

TEST_CASE("Backpropagation XOR-gate (single iteration)") {
    NeuronNetwork xorGate =
        NeuronNetwork({NeuronLayer({Neuron({0.2, -0.4}, 0), Neuron({0.7, 0.1}, 0)}),
                       NeuronLayer({Neuron({0.6, 0.9}, 0)})});

    std::pair<std::vector<double>, std::vector<double>> trainingExample;
    std::vector<double> input = {1, 1};
    std::vector<double> target = {0};
    trainingExample.first = input;
    trainingExample.second = target;

    // Test backpropagation
    xorGate.backpropagation(trainingExample);

    std::vector<std::vector<double>> expectedWeights = {
        {-0.022, 0.178, -0.422}, {-0.028, 0.672, 0.072}, {-0.146, 0.534, 0.799}};

    // Zie blz. 16 uit werkboek voor de namen
    // Neuron F
    CHECK(std::abs(xorGate.layers[0].neurons[0].bias - expectedWeights[0][0]) < 0.01);
    CHECK(std::abs(xorGate.layers[0].neurons[0].weights[0] - expectedWeights[0][1]) < 0.01);
    CHECK(std::abs(xorGate.layers[0].neurons[0].weights[1] - expectedWeights[0][2]) < 0.01);

    // Neuron G
    CHECK(std::abs(xorGate.layers[0].neurons[1].bias - expectedWeights[1][0]) < 0.01);
    CHECK(std::abs(xorGate.layers[0].neurons[1].weights[0] - expectedWeights[1][1]) < 0.01);
    CHECK(std::abs(xorGate.layers[0].neurons[1].weights[1] - expectedWeights[1][2]) < 0.01);

    // Neuron O
    CHECK(std::abs(xorGate.layers[1].neurons[0].bias - expectedWeights[2][0]) < 0.01);
    CHECK(std::abs(xorGate.layers[1].neurons[0].weights[0] - expectedWeights[2][1]) < 0.01);
    // 0.834154 != 0.799
    CHECK(std::abs(xorGate.layers[1].neurons[0].weights[1] - expectedWeights[2][2]) < 0.01);
}
