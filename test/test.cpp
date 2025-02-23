#define CATCH_CONFIG_MAIN

#include "include/catch.hpp"

#include "../src/include/neuron.hpp"
#include "../src/include/neuronlayer.hpp"
#include "../src/include/neuronnetwork.hpp"

#include <cmath>

// This test is written to test if Neuron class is written correctly before writing the network
TEST_CASE("Backpropagation AND-gate WITHOUT network") {
    // TODO: write this when network is working
    Neuron andGate = Neuron({-0.5, 0.5}, 1.5);

    std::vector<std::vector<double>> inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<bool> targets = {0, 0, 0, 1};
    std::vector<double> expectedOutputs = {0.818, 0.731, 0.881, 0.818};

    // Check if expected output is actual output (with a margin of error)
    CHECK(std::abs(andGate.output(inputs[0]) - expectedOutputs[0]) < 0.01);
    CHECK(std::abs(andGate.output(inputs[1]) - expectedOutputs[1]) < 0.01);
    CHECK(std::abs(andGate.output(inputs[2]) - expectedOutputs[2]) < 0.01);
    CHECK(std::abs(andGate.output(inputs[3]) - expectedOutputs[3]) < 0.01);

    // Check if expected errors are actual errors (iteration 1)
    double expectedError = 0.122;
    andGate.calcError(inputs[0], targets[0]);
    CHECK(std::abs(andGate.error - expectedError) < 0.01);

    // Backpropagation
    andGate.storeNewWeights(inputs[0], targets[0], inputs[0][0]);
    andGate.update();

    std::vector<double> expectedWeights = {1.378, -0.5, 0.5};

    CHECK(std::abs(andGate.bias - expectedWeights[0]) < 0.01);
    CHECK(std::abs(andGate.weights[0] - expectedWeights[1]) < 0.01);
    CHECK(std::abs(andGate.weights[1] - expectedWeights[2]) < 0.01);
}

TEST_CASE("Backpropagation AND-gate with network") {
    // TODO: write this when network is working
    NeuronNetwork andGate = NeuronNetwork({NeuronLayer({Neuron({-0.5, 0.5}, 1.5)})});

    std::vector<std::vector<double>> inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<double> targets = {0, 0, 0, 1};
    std::vector<double> expectedOutputs = {0.818, 0.731, 0.881, 0.818};

    // Check if expected output is actual output (with a margin of error)
    CHECK(std::abs(andGate.feed_forward(inputs[0])[0] - expectedOutputs[0]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[1])[0] - expectedOutputs[1]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[2])[0] - expectedOutputs[2]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[3])[0] - expectedOutputs[3]) < 0.01);

    // Backpropagation
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
