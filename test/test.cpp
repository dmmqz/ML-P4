#define CATCH_CONFIG_MAIN

#include "include/catch.hpp"

#include "../src/include/neuron.hpp"
#include "../src/include/neuronlayer.hpp"
#include "../src/include/neuronnetwork.hpp"

TEST_CASE("Test three input NOR-gate") {
    Neuron norGate = Neuron({-1, -1, -1}, 0);

    // Return true if all inputs are false
    REQUIRE(norGate.output({0, 0, 0}));
    REQUIRE(!norGate.output({1, 0, 0}));
    REQUIRE(!norGate.output({1, 1, 0}));
    REQUIRE(!norGate.output({1, 1, 1}));
    REQUIRE(!norGate.output({1, 0, 1}));
    REQUIRE(!norGate.output({0, 1, 0}));
    REQUIRE(!norGate.output({0, 1, 1}));
    REQUIRE(!norGate.output({0, 0, 1}));
}

TEST_CASE("Test Half Adder") {
    NeuronLayer layer1 = NeuronLayer(
        {Neuron({0.5, 0.5}, -0.5), Neuron({-2, -2}, 3)});
    NeuronLayer layer2 =
        NeuronLayer({Neuron({0.5, 0.5}, -1), Neuron({0, -1}, 0)});

    NeuronNetwork halfAdder = NeuronNetwork({layer1, layer2});

    // First output (sum) is XOR, second (carry) is AND
    REQUIRE(halfAdder.feed_forward({false, false}) ==
            std::vector<bool>{false, false});
    REQUIRE(halfAdder.feed_forward({true, false}) == std::vector<bool>{true, false});
    REQUIRE(halfAdder.feed_forward({false, true}) == std::vector<bool>{true, false});
    REQUIRE(halfAdder.feed_forward({true, true}) == std::vector<bool>{false, true});
}
