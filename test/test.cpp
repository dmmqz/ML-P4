#define CATCH_CONFIG_MAIN

#include "include/catch.hpp"

#include "../src/include/neuron.hpp"
#include "../src/include/neuronlayer.hpp"
#include "../src/include/neuronnetwork.hpp"

#include <cmath>

TEST_CASE("Backpropagation AND-gate (single iteration)") {
    NeuronNetwork andGate = NeuronNetwork({NeuronLayer({Neuron({-0.5, 0.5}, 1.5)})});

    std::vector<std::vector<double>> inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {0}, {0}, {1}};
    std::vector<double> expectedOutputs = {0.818, 0.731, 0.881, 0.818};

    // Check feed forward (normal sigmoid function)
    CHECK(std::abs(andGate.feed_forward(inputs[0])[0] - expectedOutputs[0]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[1])[0] - expectedOutputs[1]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[2])[0] - expectedOutputs[2]) < 0.01);
    CHECK(std::abs(andGate.feed_forward(inputs[3])[0] - expectedOutputs[3]) < 0.01);

    // Test backpropagation
    andGate.backpropagation(inputs[0], targets[0]);

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

    std::vector<double> input = {1, 1};
    std::vector<double> target = {0};

    // Test backpropagation
    xorGate.backpropagation(input, target);

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
    CHECK(std::abs(xorGate.layers[1].neurons[0].weights[1] - expectedWeights[2][2]) < 0.01);
}

TEST_CASE("Backpropagation Half Adder (single iteration)") {
    NeuronNetwork halfAdder = NeuronNetwork(
        {NeuronLayer({Neuron({0, 0.1}, 0), Neuron({0.2, 0.3}, 0), Neuron({0.4, 0.5}, 0)}),
         NeuronLayer({Neuron({0.6, 0.7, 0.8}, 0), Neuron({0.9, 1, 1.1}, 0)})});

    halfAdder.backpropagation({1, 1}, {0, 1});

    std::vector<std::vector<double>> expectedWeights = {{-0.016, -0.016, 0.084},
                                                        {-0.018, 0.182, 0.282},
                                                        {-0.018, 0.382, 0.482},
                                                        {-0.131, 0.531, 0.618, 0.707},
                                                        {0.015, 0.908, 1.008, 1.111}};

    // Zie blz. 22-36 uit het werkboek (ik heb het zelf uitgewerkt)
    // First layer
    // Neuron F
    CHECK(std::abs(halfAdder.layers[0].neurons[0].bias - expectedWeights[0][0]) < 0.01);
    CHECK(std::abs(halfAdder.layers[0].neurons[0].weights[0] - expectedWeights[0][1]) < 0.01);
    CHECK(std::abs(halfAdder.layers[0].neurons[0].weights[1] - expectedWeights[0][2]) < 0.01);

    // Neuron G
    CHECK(std::abs(halfAdder.layers[0].neurons[1].bias - expectedWeights[1][0]) < 0.01);
    CHECK(std::abs(halfAdder.layers[0].neurons[1].weights[0] - expectedWeights[1][1]) < 0.01);
    CHECK(std::abs(halfAdder.layers[0].neurons[1].weights[1] - expectedWeights[1][2]) < 0.01);

    // Neuron H
    CHECK(std::abs(halfAdder.layers[0].neurons[2].bias - expectedWeights[2][0]) < 0.01);
    CHECK(std::abs(halfAdder.layers[0].neurons[2].weights[0] - expectedWeights[2][1]) < 0.01);
    CHECK(std::abs(halfAdder.layers[0].neurons[2].weights[1] - expectedWeights[2][2]) < 0.01);

    // Second Layer
    // Neuron S
    CHECK(std::abs(halfAdder.layers[1].neurons[0].bias - expectedWeights[3][0]) < 0.01);
    CHECK(std::abs(halfAdder.layers[1].neurons[0].weights[0] - expectedWeights[3][1]) < 0.01);
    CHECK(std::abs(halfAdder.layers[1].neurons[0].weights[1] - expectedWeights[3][2]) < 0.01);
    CHECK(std::abs(halfAdder.layers[1].neurons[0].weights[2] - expectedWeights[3][3]) < 0.01);

    // Neuron C
    CHECK(std::abs(halfAdder.layers[1].neurons[1].bias - expectedWeights[4][0]) < 0.01);
    CHECK(std::abs(halfAdder.layers[1].neurons[1].weights[0] - expectedWeights[4][1]) < 0.01);
    CHECK(std::abs(halfAdder.layers[1].neurons[1].weights[1] - expectedWeights[4][2]) < 0.01);
    CHECK(std::abs(halfAdder.layers[1].neurons[1].weights[2] - expectedWeights[4][3]) < 0.01);
}

TEST_CASE("Train AND-gate") {
    NeuronNetwork andGate = NeuronNetwork({NeuronLayer({Neuron({-0.5, 0.5}, 1.5)})});

    std::vector<std::vector<double>> inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {0}, {0}, {1}};

    andGate.train(inputs, targets);

    CHECK(andGate.feed_forward(inputs[0])[0] < 0.5);
    CHECK(andGate.feed_forward(inputs[1])[0] < 0.5);
    CHECK(andGate.feed_forward(inputs[2])[0] < 0.5);
    CHECK(andGate.feed_forward(inputs[3])[0] >= 0.5);
}

TEST_CASE("Train XOR-gate") {
    NeuronNetwork xorGate =
        NeuronNetwork({NeuronLayer({Neuron({0.2, -0.4}, 0), Neuron({0.7, 0.1}, 0)}),
                       NeuronLayer({Neuron({0.6, 0.9}, 0)})});

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0}, {1}, {1}, {0}};

    xorGate.train(inputs, targets);

    CHECK(xorGate.feed_forward(inputs[0])[0] < 0.5);
    CHECK(xorGate.feed_forward(inputs[1])[0] >= 0.5);
    CHECK(xorGate.feed_forward(inputs[2])[0] >= 0.5);
    CHECK(xorGate.feed_forward(inputs[3])[0] < 0.5);
}

TEST_CASE("Train Half Adder") {
    NeuronNetwork halfAdder = NeuronNetwork(
        {NeuronLayer({Neuron({0, 0.1}, 0), Neuron({0.2, 0.3}, 0), Neuron({0.4, 0.5}, 0)}),
         NeuronLayer({Neuron({0.6, 0.7, 0.8}, 0), Neuron({0.9, 1, 1.1}, 0)})});

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> targets = {{0, 0}, {1, 0}, {1, 0}, {0, 1}};

    halfAdder.train(inputs, targets);

    std::vector<std::vector<double>> outputs{};
    for (const std::vector<double> input : inputs) {
        outputs.push_back(halfAdder.feed_forward(input));
    }

    CHECK((outputs[0][0] < 0.5 && outputs[0][1] < 0.5));
    CHECK((outputs[1][0] >= 0.5 && outputs[1][1] < 0.5));
    CHECK((outputs[2][0] >= 0.5 && outputs[2][1] < 0.5));
    CHECK((outputs[3][0] < 0.5 && outputs[3][1] >= 0.5));
}
