#define CATCH_CONFIG_MAIN

#include "include/catch.hpp"

#include "../src/include/neuron.hpp"
#include "../src/include/neuronlayer.hpp"
#include "../src/include/neuronnetwork.hpp"

#include <cmath>

TEST_CASE("Test INVERT-, AND- and -OR-gates") {
    Neuron invertGate = Neuron({-1}, 0);
    REQUIRE(invertGate.output({0}) >= 0.5);
    REQUIRE(invertGate.output({1}) < 0.5);

    Neuron orGate = Neuron({0.5, 0.5}, -0.5);
    REQUIRE(orGate.output({0, 0}) < 0.5);
    REQUIRE(orGate.output({1, 0}) >= 0.5);
    REQUIRE(orGate.output({0, 1}) >= 0.5);
    REQUIRE(orGate.output({1, 1}) >= 0.5);

    Neuron andGate = Neuron({0.5, 0.5}, -0.6);
    REQUIRE(andGate.output({0, 0}) < 0.5);
    REQUIRE(andGate.output({1, 0}) < 0.5);
    REQUIRE(andGate.output({0, 1}) < 0.5);
    REQUIRE(andGate.output({1, 1}) >= 0.5);
}

TEST_CASE("Test three input NOR-gate") {
    Neuron norGate = Neuron({-100, -100, -100}, 0);

    // Return true if all inputs are false
    REQUIRE(norGate.output({0, 0, 0}) >= 0.5);
    REQUIRE(!(norGate.output({1, 0, 0}) >= 0.5));
    REQUIRE(!(norGate.output({1, 1, 0}) >= 0.5));
    REQUIRE(!(norGate.output({1, 1, 1}) >= 0.5));
    REQUIRE(!(norGate.output({1, 0, 1}) >= 0.5));
    REQUIRE(!(norGate.output({0, 1, 0}) >= 0.5));
    REQUIRE(!(norGate.output({0, 1, 1}) >= 0.5));
    REQUIRE(!(norGate.output({0, 0, 1}) >= 0.5));
}

TEST_CASE("Test Half Adder") {
    NeuronLayer layer1 = NeuronLayer({Neuron({10, 10}, -5), Neuron({10, 10}, -15)});
    NeuronLayer layer2 = NeuronLayer({Neuron({10, -10}, -1), Neuron({10, 10}, -15)});

    NeuronNetwork halfAdder = NeuronNetwork({layer1, layer2});

    // First output (sum) is XOR, second (carry) is AND
    REQUIRE(halfAdder.feed_forward({0, 0})[0] < 0.5);
    REQUIRE(halfAdder.feed_forward({0, 0})[1] < 0.5);

    REQUIRE(halfAdder.feed_forward({1, 0})[0] >= 0.5);
    REQUIRE(halfAdder.feed_forward({1, 0})[1] < 0.5);

    REQUIRE(halfAdder.feed_forward({0, 1})[0] >= 0.5);
    REQUIRE(halfAdder.feed_forward({0, 1})[1] < 0.5);

    REQUIRE(halfAdder.feed_forward({1, 1})[0] < 0.5);
    REQUIRE(halfAdder.feed_forward({1, 1})[1] >= 0.5);
}

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
    double error = andGate.error(inputs[0], targets[0]);
    CHECK(std::abs(error - expectedError) < 0.01);

    // Backpropagation
    andGate.update(inputs[0], targets[0], inputs[0][0]);

    std::vector<double> expectedWeights = {1.378, -0.5, 0.5};

    CHECK(std::abs(andGate.bias - expectedWeights[0]) < 0.01);
    CHECK(std::abs(andGate.weights[0] - expectedWeights[1]) < 0.01);
    CHECK(std::abs(andGate.weights[1] - expectedWeights[2]) < 0.01);
}
