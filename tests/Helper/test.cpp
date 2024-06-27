#include <catch2/catch_all.hpp>
#include "catch_config.hpp"
#include "NNHelper.h"

#include <cmath>
#include <limits>

bool approxEqual(double a, double b, double epsilon = std::numeric_limits<double>::epsilon())
{
    return std::abs(a - b) <= epsilon * std::max(std::abs(a), std::abs(b));
}

TEST_CASE("Random Generator Test", "[randomGenerator]")
{
    SECTION("Random Number Generation Test")
    {
        double minValue = 0.0;
        double maxValue = 1.0;
        double randNum = randomGenerator(minValue, maxValue);
        REQUIRE(randNum >= minValue);
        REQUIRE(randNum <= maxValue);
    }
}

TEST_CASE("Operator Overloading Test", "[operatorOverloading]")
{
    SECTION("Matrix-Vector Multiplication Test")
    {
        // Test Matrix and Vector
        std::vector<std::vector<double>> matrix = {{1, 2}, {3, 4}};
        std::vector<double> vector = {1, 2};
        std::vector<double> result = matrix * vector;
        std::vector<double> expected = {5, 11};
        REQUIRE(result == expected);
    }

    SECTION("Matrix-Matrix Multiplication Test")
    {
        // Test Matrices
        std::vector<std::vector<double>> matrix1 = {{1, 2}, {3, 4}};
        std::vector<std::vector<double>> matrix2 = {{1, 2}, {3, 4}};
        std::vector<std::vector<double>> result = matrix1 * matrix2;
        std::vector<std::vector<double>> expected = {{7, 10}, {15, 22}};
        REQUIRE(result == expected);
    }
}

TEST_CASE("Matrix Functions Test", "[matrixFunctions]")
{
    SECTION("Transpose Test")
    {
        // Test Matrix
        std::vector<std::vector<double>> matrix = {{1, 2, 3}, {4, 5, 6}};
        std::vector<std::vector<double>> result = transpose(matrix);
        std::vector<std::vector<double>> expected = {{1, 4}, {2, 5}, {3, 6}};
        REQUIRE(result == expected);
    }

    SECTION("Diagonal Matrix Creation Test")
    {
        // Test Vector
        std::vector<double> vector = {1, 2, 3};
        std::vector<std::vector<double>> result = createDiagonalMatrix(vector);
        std::vector<std::vector<double>> expected = {{1, 0, 0}, {0, 2, 0}, {0, 0, 3}};
        REQUIRE(result == expected);
    }
}

TEST_CASE("Utility Functions Test", "[utilityFunctions]")
{
    SECTION("Clip Values Test")
    {
        // Test Matrix
        std::vector<std::vector<double>> matrix = {{0.5, 1.5}, {0, 1}};
        std::vector<std::vector<double>> result = clipValues(matrix);
        std::vector<std::vector<double>> expected = {{0.5, 1.0}, {0.0, 1.0}};
        printMatrix(result);

        REQUIRE(approxEqual(result[0][0], expected[0][0], 1)); // Compare with tolerance
        REQUIRE(approxEqual(result[0][1], expected[0][1], 1)); // Compare with tolerance
        REQUIRE(approxEqual(result[1][0], expected[1][0], 1)); // Compare with tolerance
        REQUIRE(approxEqual(result[1][1], expected[1][1], 1)); // Compare with tolerance
    }

    SECTION("Sum Elements On Axis Zero Test")
    {
        // Test Matrix
        std::vector<std::vector<double>> matrix = {{1, 2, 3}, {4, 5, 6}};
        std::vector<double> result = sumElementsOnAxisZero(matrix);
        std::vector<double> expected = {5, 7, 9};
        REQUIRE(result == expected);
    }
}

TEST_CASE("randomGenerator generates values within the range", "[randomGenerator]")
{
    double minValue = 0.0;
    double maxValue = 1.0;
    for (int i = 0; i < 100; ++i)
    {
        double value = randomGenerator(minValue, maxValue);
        REQUIRE(value >= minValue);
        REQUIRE(value <= maxValue);
    }
}

TEST_CASE("createIdentityMatrix creates correct identity matrix", "[createIdentityMatrix]")
{
    int size = 3;
    std::vector<std::vector<double>> expected = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}};
    std::vector<std::vector<double>> result = createIdentityMatrix(size);
    REQUIRE(result == expected);
}

// Add more tests for other functions...

TEST_CASE("matrix-vector multiplication", "[operator*]")
{
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<double> vector = {1.0, 1.0};
    std::vector<double> expected = {3.0, 7.0};
    std::vector<double> result = matrix * vector;
    REQUIRE(result == expected);
}

TEST_CASE("matrix-matrix multiplication", "[operator*]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> matrix2 = {
        {2.0, 0.0},
        {1.0, 2.0}};
    std::vector<std::vector<double>> expected = {
        {4.0, 4.0},
        {10.0, 8.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result == expected);
}

TEST_CASE("scalar-matrix multiplication", "[operator*]")
{
    double scalar = 2.0;
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> expected = {
        {2.0, 4.0},
        {6.0, 8.0}};
    std::vector<std::vector<double>> result = scalar * matrix;
    REQUIRE(result == expected);
}

TEST_CASE("scalar-vector multiplication", "[operator*]")
{
    double scalar = 2.0;
    std::vector<double> vector = {1.0, 2.0};
    std::vector<double> expected = {2.0, 4.0};
    std::vector<double> result = scalar * vector;
    REQUIRE(result == expected);
}

// TEST_CASE("matrix-matrix division", "[operator/]")
// {
//     std::vector<std::vector<double>> matrix1 = {
//         {2.0, 4.0},
//         {6.0, 8.0}};
//     std::vector<std::vector<double>> matrix2 = {
//         {2.0, 2.0},
//         {3.0, 4.0}};
//     std::vector<std::vector<double>> expected = {
//         {1.0, 2.0},
//         {2.0, 2.0}};
//     std::vector<std::vector<double>> result = matrix1 / matrix2;
//     REQUIRE(result == expected);
// }

TEST_CASE("matrix-scalar division", "[operator/]")
{
    std::vector<std::vector<double>> matrix = {
        {2.0, 4.0},
        {6.0, 8.0}};
    int scalar = 2;
    std::vector<std::vector<double>> expected = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> result = matrix / scalar;
    REQUIRE(result == expected);
}

TEST_CASE("vector addition", "[operator+]")
{
    std::vector<double> vec = {1.0, 2.0};
    std::vector<double> values = {3.0, 4.0};
    std::vector<double> expected = {4.0, 6.0};
    std::vector<double> result = vec + values;
    REQUIRE(result == expected);
}

TEST_CASE("matrix-vector addition", "[operator+]")
{
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<double> vector = {1.0, 1.0};
    std::vector<std::vector<double>> expected = {
        {2.0, 3.0},
        {4.0, 5.0}};
    std::vector<std::vector<double>> result = matrix + vector;
    REQUIRE(result == expected);
}

TEST_CASE("matrix-matrix addition", "[operator+]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> matrix2 = {
        {2.0, 1.0},
        {0.0, 1.0}};
    std::vector<std::vector<double>> expected = {
        {3.0, 3.0},
        {3.0, 5.0}};
    std::vector<std::vector<double>> result = matrix1 + matrix2;
    REQUIRE(result == expected);
}

TEST_CASE("transpose matrix", "[transpose]")
{
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> expected = {
        {1.0, 3.0},
        {2.0, 4.0}};
    std::vector<std::vector<double>> result = transpose(matrix);
    REQUIRE(result == expected);
}

TEST_CASE("sum elements on axis zero", "[sumElementsOnAxisZero]")
{
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<double> expected = {4.0, 6.0};
    std::vector<double> result = sumElementsOnAxisZero(matrix);
    REQUIRE(result == expected);
}

TEST_CASE("get number of rows", "[getNumOfRows]")
{
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}};
    int expected = 2;
    int result = getNumOfRows(matrix);
    REQUIRE(result == expected);
}

TEST_CASE("get number of columns", "[getNumOfColumns]")
{
    std::vector<std::vector<double>> matrix = {
        {1.0, 2.0},
        {3.0, 4.0}};
    int expected = 2;
    int result = getNumOfColumns(matrix);
    REQUIRE(result == expected);
}

TEST_CASE("clip values", "[clipValues]")
{
    std::vector<std::vector<double>> matrix = {
        {0.5, 1.5},
        {-0.5, 0.0}};
    std::vector<std::vector<double>> expected = {
        {0.5, 1.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> result = clipValues(matrix);
    REQUIRE(approxEqual(result[0][0], expected[0][0], 1)); // Compare with tolerance
    REQUIRE(approxEqual(result[0][1], expected[0][1], 1)); // Compare with tolerance
    REQUIRE(approxEqual(result[1][0], expected[1][0], 1)); // Compare with tolerance
    REQUIRE(approxEqual(result[1][1], expected[1][1], 1)); // Compare with tolerance
}

TEST_CASE("create diagonal matrix", "[createDiagonalMatrix]")
{
    std::vector<double> vector = {1.0, 2.0, 3.0};
    std::vector<std::vector<double>> expected = {
        {1.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 3.0}};
    std::vector<std::vector<double>> result = createDiagonalMatrix(vector);
    REQUIRE(result == expected);
}

TEST_CASE("create identity matrix", "[createIdentityMatrix]")
{
    int size = 3;
    std::vector<std::vector<double>> expected = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}};
    std::vector<std::vector<double>> result = createIdentityMatrix(size);
    REQUIRE(result == expected);
}

TEST_CASE("sLToOneHotEncodedL", "[sLToOneHotEncodedL]")
{
    std::vector<int> yTrue = {0, 1, 2};
    int labels = 3;
    std::vector<std::vector<double>> expected = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0}};
    std::vector<std::vector<double>> result = sLToOneHotEncodedL(yTrue, labels);
    REQUIRE(result == expected);
}

TEST_CASE("oneHotEncodedToDiscrete", "[oneHotEncodedToDiscrete]")
{
    std::vector<std::vector<int>> yTrue = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}};
    std::vector<int> expected = {0, 1, 2};
    std::vector<int> result = oneHotEncodedToDiscrete(yTrue);
    REQUIRE(result == expected);
}

TEST_CASE("Matrix multiplication with compatible dimensions", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> matrix2 = {
        {2.0, 0.0},
        {1.0, 2.0}};
    std::vector<std::vector<double>> expected = {
        {4.0, 4.0},
        {10.0, 8.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result == expected);
}

TEST_CASE("Matrix multiplication with incompatible dimensions", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}};
    std::vector<std::vector<double>> matrix2 = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> expected = {
        {0.0, 0.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result == expected);
}

TEST_CASE("Matrix multiplication with zero matrices2", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {0.0, 0.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> matrix2 = {
        {0.0, 0.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> expected = {
        {0.0, 0.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result == expected);
}

TEST_CASE("Matrix multiplication with identity matrix", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 0.0},
        {0.0, 1.0}};
    std::vector<std::vector<double>> matrix2 = {
        {2.0, 3.0},
        {4.0, 5.0}};
    std::vector<std::vector<double>> expected = {
        {2.0, 3.0},
        {4.0, 5.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result == expected);
}

TEST_CASE("Matrix multiplication with negative values", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, -2.0},
        {-3.0, 4.0}};
    std::vector<std::vector<double>> matrix2 = {
        {-2.0, 0.0},
        {1.0, -2.0}};
    std::vector<std::vector<double>> expected = {
        {-4.0, 4.0},
        {10.0, -8.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result == expected);
}

TEST_CASE("Matrix multiplication with compatible dimensions2", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> matrix2 = {
        {2.0, 0.0},
        {1.0, 2.0}};
    std::vector<std::vector<double>> expected = {
        {4.0, 4.0},
        {10.0, 8.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result.size() == expected.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        REQUIRE(result[i].size() == expected[i].size());
        for (size_t j = 0; j < result[i].size(); ++j)
        {
            REQUIRE(approxEqual(result[i][j], expected[i][j]));
        }
    }
}

TEST_CASE("Matrix multiplication with incompatible dimensions2", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}};
    std::vector<std::vector<double>> matrix2 = {
        {1.0, 2.0},
        {3.0, 4.0}};
    std::vector<std::vector<double>> expected = {
        {0.0, 0.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result == expected);
}

TEST_CASE("Matrix multiplication with zero matrices", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {0.0, 0.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> matrix2 = {
        {0.0, 0.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> expected = {
        {0.0, 0.0},
        {0.0, 0.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result.size() == expected.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        REQUIRE(result[i].size() == expected[i].size());
        for (size_t j = 0; j < result[i].size(); ++j)
        {
            REQUIRE(approxEqual(result[i][j], expected[i][j]));
        }
    }
}

TEST_CASE("Matrix multiplication with identity matrix2", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, 0.0},
        {0.0, 1.0}};
    std::vector<std::vector<double>> matrix2 = {
        {2.0, 3.0},
        {4.0, 5.0}};
    std::vector<std::vector<double>> expected = {
        {2.0, 3.0},
        {4.0, 5.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result.size() == expected.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        REQUIRE(result[i].size() == expected[i].size());
        for (size_t j = 0; j < result[i].size(); ++j)
        {
            REQUIRE(approxEqual(result[i][j], expected[i][j]));
        }
    }
}

TEST_CASE("Matrix multiplication with negative values2", "[matrix_multiplication]")
{
    std::vector<std::vector<double>> matrix1 = {
        {1.0, -2.0},
        {-3.0, 4.0}};
    std::vector<std::vector<double>> matrix2 = {
        {-2.0, 0.0},
        {1.0, -2.0}};
    std::vector<std::vector<double>> expected = {
        {-4.0, 4.0},
        {10.0, -8.0}};
    std::vector<std::vector<double>> result = matrix1 * matrix2;
    REQUIRE(result.size() == expected.size());
    for (size_t i = 0; i < result.size(); ++i)
    {
        REQUIRE(result[i].size() == expected[i].size());
        for (size_t j = 0; j < result[i].size(); ++j)
        {
            REQUIRE(approxEqual(result[i][j], expected[i][j]));
        }
    }
}