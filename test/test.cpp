#define CATCH_CONFIG_MAIN

#include "include/catch.hpp"

#include "../src/include/neuron.hpp"
#include "../src/include/neuronlayer.hpp"
#include "../src/include/neuronnetwork.hpp"

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
