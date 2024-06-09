#include <catch2/catch_all.hpp>
#include "NNLayer.h"

// Test cases for NN_Layer_Dense class
TEST_CASE("NN_Layer_Dense class tests", "[NN_Layer_Dense]")
{
    SECTION("Initialization")
    {
        // Test case 1: Verify weights and biases initialization
        NN_Layer_Dense layer1(2, 3);
        auto weights1 = layer1.getWeights();
        auto bias1 = layer1.getBias();
        REQUIRE(weights1.size() == 2);
        REQUIRE(weights1[0].size() == 3);
        REQUIRE(bias1.size() == 3);

        // Test case 2: Verify weights and biases initialization for a different configuration
        NN_Layer_Dense layer2(3, 4);
        auto weights2 = layer2.getWeights();
        auto bias2 = layer2.getBias();
        REQUIRE(weights2.size() == 3);
        REQUIRE(weights2[0].size() == 4);
        REQUIRE(bias2.size() == 4);
    }

    SECTION("Forward propagation")
    {
        // Test case 3: Verify forward propagation for a simple input
        NN_Layer_Dense layer(2, 2);
        std::vector<std::vector<double>> input = {{1.0, 2.0}};
        auto output = layer.forward(input);
        REQUIRE(output.size() == 1);
        REQUIRE(output[0].size() == 2);

        // Test case 4: Verify forward propagation for another input
        input = {{-1.0, 3.0}};
        output = layer.forward(input);
        REQUIRE(output.size() == 1);
        REQUIRE(output[0].size() == 2);
    }
}

TEST_CASE("NN_Layer_Dense class tests", "Backward")
{
    // Test case: Verify backward propagation for a simple gradient
    NN_Layer_Dense layer(2, 2);
    printMatrix(layer.getdInput());
    printMatrix(layer.getWeights());
    printVector(layer.getBias());
    std::vector<std::vector<double>> gradient = {{0.1, 0.2},
                                                 {10.0, 8.0}};
    layer.backward(gradient);
    // REQUIRE(dInput.size() == 1);
    // REQUIRE(dInput[0].size() == 2);
}

