#include "NNActivation.h"

std::vector<std::vector<double>> NN_ActivationReLU::forward(std::vector<std::vector<double>> &matrix)
{
    for (auto &&row : matrix)
    {
        for (auto &&el : row)
        {
            if (el < 0)
            {
                el = 0;
            }
        }
    }

    outputValue = matrix;
    return matrix;
}

std::vector<std::vector<double>> NN_ActivationSoftMax::forward(std::vector<std::vector<double>> &matrix)
{
    int n = getNumOfRows(matrix);
    int m = getNumOfColumns(matrix);
    double maxVal = -std::numeric_limits<double>::infinity(); // why and how is that working?

    std::vector<std::vector<double>> outputMatrix(n, std::vector<double>(m));

    // Find the maximum value
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            if (matrix[i][j] > maxVal)
                maxVal = matrix[i][j];
        }
    }

    // Next calculate the overall sum of all the element (which values have been exponentiated)
    for (int i = 0; i < n; i++)
    {
        double expSum = 0; // Reset expSum for each row
        for (int j = 0; j < m; j++)
        {
            outputMatrix[i][j] = std::exp(matrix[i][j] - maxVal);
            expSum += outputMatrix[i][j];
        }

        // Finally we divide each element of the matrix by the calculated expSum.
        for (int j = 0; j < m; j++)
        {
            outputMatrix[i][j] = outputMatrix[i][j] / expSum;
        }
    }

    outputValue = outputMatrix;
    return outputMatrix;
}