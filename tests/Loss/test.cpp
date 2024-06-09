#include <catch2/catch_all.hpp>
#include "catch_config.hpp"
#include "NNLoss.h"

#include <cmath>
#include <limits>

bool approxEqual(double a, double b, double epsilon = std::numeric_limits<double>::epsilon())
{
    return std::abs(a - b) <= epsilon * std::max(std::abs(a), std::abs(b));
}

TEST_CASE("Loss Calculation with NN_CategoricalCrossEntropyLoss", "[NN_Loss]")
{
    NN_CategoricalCrossEntropyLoss lossFunction;

    SECTION("Simple case with one sample")
    {
        std::vector<std::vector<double>> predicted = {{0.1, 0.9}};
        std::vector<int> trueValues = {1};

        double loss = lossFunction.calculate(predicted, trueValues);
        REQUIRE(approxEqual(loss, -std::log(0.9)));
    }

    SECTION("Simple case with multiple samples")
    {
        std::vector<std::vector<double>> predicted = {
            {0.1, 0.9},
            {0.8, 0.2}};
        std::vector<int> trueValues = {1, 0};

        double loss = lossFunction.calculate(predicted, trueValues);
        double expectedLoss = (-std::log(0.9) - std::log(0.8)) / 2;
        REQUIRE(approxEqual(loss, expectedLoss));
    }

    SECTION("Clipping values to prevent log(0)")
    {
        std::vector<std::vector<double>> predicted = {{1e-10, 1.0 - 1e-10}};
        std::vector<int> trueValues = {0};

        double loss = lossFunction.calculate(predicted, trueValues);
        double expected_loss = -std::log(1e-10);

        // Define a small tolerance value
        double tolerance = 1e-6; // Adjust this as needed

        // Check if the absolute difference between loss and expected_loss is within tolerance
        REQUIRE(std::abs(loss - expected_loss) < tolerance);
    }

    SECTION("Different batch sizes")
    {
        std::vector<std::vector<double>> predicted = {
            {0.1, 0.9},
            {0.8, 0.2},
            {0.4, 0.6}};
        std::vector<int> trueValues = {1, 0, 1};

        double loss = lossFunction.calculate(predicted, trueValues);
        double expectedLoss = (-std::log(0.9) - std::log(0.8) - std::log(0.6)) / 3;
        REQUIRE(approxEqual(loss, expectedLoss));
    }
}

TEST_CASE("Gradient Calculation with NN_CategoricalCrossEntropyLoss", "[NN_Loss]")
{
    NN_CategoricalCrossEntropyLoss lossFunction;

    SECTION("Simple case with one sample")
    {
        std::vector<std::vector<double>> dValues = {{0.1, 0.9}};
        std::vector<int> trueValues = {1};

        auto gradients = lossFunction.backward(dValues, trueValues);
        std::vector<std::vector<double>> expectedGradients = {{0.0, -1.0}};
        REQUIRE(gradients == expectedGradients);
    }

    SECTION("Multiple samples")
    {
        std::vector<std::vector<double>> dValues = {
            {0.1, 0.9},
            {0.8, 0.2}};
        std::vector<int> trueValues = {1, 0};

        auto gradients = lossFunction.backward(dValues, trueValues);
        std::vector<std::vector<double>> expectedGradients = {
            {0.0, -1.0},
            {-1.0, 0.0}};
        REQUIRE(gradients == expectedGradients);
    }

    SECTION("Gradients with clipping")
    {
        std::vector<std::vector<double>> dValues = {{1e-10, 1.0 - 1e-10}};
        std::vector<int> trueValues = {0};

        auto gradients = lossFunction.backward(dValues, trueValues);
        REQUIRE(gradients[0][0] < 0);
        REQUIRE(gradients[0][1] > 0);
    }

    SECTION("Different batch sizes")
    {
        std::vector<std::vector<double>> dValues = {
            {0.1, 0.9},
            {0.8, 0.2},
            {0.4, 0.6}};
        std::vector<int> trueValues = {1, 0, 1};

        auto gradients = lossFunction.backward(dValues, trueValues);
        std::vector<std::vector<double>> expectedGradients = {
            {0.0, -1.0 / 3.0},
            {-1.0 / 3.0, 0.0},
            {0.0, -1.0 / 3.0}};
        REQUIRE(gradients == expectedGradients);
    }
}

TEST_CASE("Loss Calculation with NN_ActivationSMaxCategoricalCrossEntropyLoss", "[NN_Loss]")
{
    NN_ActivationSMaxCategoricalCrossEntropyLoss lossFunction;

    SECTION("Simple case with one sample")
    {
        std::vector<std::vector<double>> predicted = {{0.1, 0.9}};
        std::vector<int> trueValues = {1};

        double loss = lossFunction.forward(predicted, trueValues)[0];
        REQUIRE(approxEqual(loss, -std::log(0.9)));
    }

    SECTION("Simple case with multiple samples")
    {
        std::vector<std::vector<double>> predicted = {
            {0.1, 0.9},
            {0.8, 0.2}};
        std::vector<int> trueValues = {1, 0};

        double loss = lossFunction.forward(predicted, trueValues)[0];
        double expectedLoss = (-std::log(0.9) - std::log(0.8)) / 2;
        REQUIRE(approxEqual(loss, expectedLoss));
    }

    SECTION("Edge case with zero probabilities")
    {
        std::vector<std::vector<double>> predicted = {{0.0, 1.0}};
        std::vector<int> trueValues = {0};

        REQUIRE_THROWS_AS(lossFunction.calculate(predicted, trueValues), std::domain_error);
    }

    SECTION("Clipping values to prevent log(0)")
    {
        std::vector<std::vector<double>> predicted = {{1e-10, 1.0 - 1e-10}};
        std::vector<int> trueValues = {0};

        double loss = lossFunction.forward(predicted, trueValues)[0];
        REQUIRE(approxEqual(loss, -std::log(1e-10)));
    }

    SECTION("Different batch sizes")
    {
        std::vector<std::vector<double>> predicted = {
            {0.1, 0.9},
            {0.8, 0.2},
            {0.4, 0.6}};
        std::vector<int> trueValues = {1, 0, 1};

        double loss = lossFunction.forward(predicted, trueValues)[0];
        double expectedLoss = (-std::log(0.9) - std::log(0.8) - std::log(0.6)) / 3;
        REQUIRE(approxEqual(loss, expectedLoss));
    }
}

TEST_CASE("Gradient Calculation with NN_ActivationSMaxCategoricalCrossEntropyLoss", "[NN_Loss]")
{
    NN_ActivationSMaxCategoricalCrossEntropyLoss lossFunction;

    SECTION("Simple case with one sample")
    {
        std::vector<std::vector<double>> dValues = {{0.1, 0.9}};
        std::vector<int> trueValues = {1};

        auto gradients = lossFunction.backward(dValues, trueValues);
        std::vector<std::vector<double>> expectedGradients = {{0.1, -0.1}};
        REQUIRE(gradients == expectedGradients);
    }

    SECTION("Multiple samples")
    {
        std::vector<std::vector<double>> dValues = {
            {0.1, 0.9},
            {0.8, 0.2}};
        std::vector<int> trueValues = {1, 0};

        auto gradients = lossFunction.backward(dValues, trueValues);
        std::vector<std::vector<double>> expectedGradients = {
            {0.1, -0.1},
            {-0.1, 0.1}};
        REQUIRE(gradients == expectedGradients);
    }

    SECTION("Gradients with clipping")
    {
        std::vector<std::vector<double>> dValues = {{1e-10, 1.0 - 1e-10}};
        std::vector<int> trueValues = {0};

        auto gradients = lossFunction.backward(dValues, trueValues);
        REQUIRE(gradients[0][0] < 0);
        REQUIRE(gradients[0][1] > 0);
    }

    SECTION("Different batch sizes")
    {
        std::vector<std::vector<double>> dValues = {
            {0.1, 0.9},
            {0.8, 0.2},
            {0.4, 0.6}};
        std::vector<int> trueValues = {1, 0, 1};

        auto gradients = lossFunction.backward(dValues, trueValues);
        std::vector<std::vector<double>> expectedGradients = {
            {0.1, -0.1},
            {-0.1, 0.1},
            {0.133333, -0.133333}};
        REQUIRE(gradients == expectedGradients);
    }
}