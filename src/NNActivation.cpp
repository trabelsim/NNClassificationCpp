#include "NNActivation.h"
#include <iostream>
#include <algorithm>
#include <cmath>

std::vector<std::vector<double>> NN_ActivationReLU::forward(std::vector<std::vector<double>> &matrix)
{
    // Remember input values
    std::vector<std::vector<double>> relOutput = matrix;

    for (auto &row : relOutput)
    {
        for (auto &el : row)
        {
            el = std::max(0.0, el);
        }
    }

    output_ = relOutput;
    return relOutput;
}

std::vector<std::vector<double>>& NN_ActivationReLU::backward(std::vector<std::vector<double>> &dValues)
{
    dInput_ = dValues;
    for (auto &&row : dInput_)
    {
        for (auto &&el : row)
        {
            if (el <= 0)
            {
                el = 0.0;
            }
        }
    }

    return dInput_;
}

std::vector<std::vector<double>> NN_ActivationSoftMax::forward(std::vector<std::vector<double>> &matrix)
{
    input_ = matrix;
    int n = getNumOfRows(input_);
    int m = getNumOfColumns(input_);

    std::vector<std::vector<double>> outputMatrix(n, std::vector<double>(m));

    // Next calculate the overall sum of all the element (which values have been exponentiated)
    for (int i = 0; i < n; i++)
    {
        double maxVal = *std::max_element(input_[i].begin(), input_[i].end());
        double expSum = 0; // Reset expSum for each row

        for (int j = 0; j < m; j++)
        {
            outputMatrix[i][j] = std::exp(input_[i][j] - maxVal);
            expSum += outputMatrix[i][j];
        }

        // Finally we divide each element of the matrix by the calculated expSum.
        for (int j = 0; j < m; j++)
        {
            outputMatrix[i][j] /=  expSum;
        }
    }

    output_ = outputMatrix;
    return outputMatrix;
}

//TODO KUKU
std::vector<std::vector<double>> NN_ActivationSoftMax::backward(std::vector<std::vector<double>> &matrix)
{
    if (!((matrix.size() == dInput_.size()) && (matrix[0].size() == dInput_[0].size())))
    {
        std::cout << "Error - find a way to handle backward with different matrix shape." << std::endl;
        return matrix;
    }

    dInput_ = std::vector<std::vector<double>>(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));

    // Iterate through each sample
    for (int sample = 0; sample < output_.size(); sample++)
    {
        //  extract output and gradient's sample
        std::vector<double> singleOutput = output_[sample];
        std::vector<double> singledValues = matrix[sample];

        // Reshape output to a column vector:
        std::vector<std::vector<double>> singleOutputCol(singleOutput.size(), std::vector<double>(1));
        for (int i = 0; i < singleOutputCol.size(); i++)
        {
            singleOutputCol[i][0] = singleOutput[i];
        }

        // Jacobian calculation
        std::vector<std::vector<double>> jacobianMatrix = createDiagonalMatrix(singleOutput);
        std::vector<std::vector<double>> outerProduct = singleOutputCol * transpose(singleOutputCol);

        for(int i = 0; i < jacobianMatrix.size(); i++)
        {
            for(int j=0; j < jacobianMatrix[i].size(); j++)
            {
                jacobianMatrix[i][j] -= outerProduct[i][j];
            }
        }

        // Convert the dSingleValues to a column vector as well
        std::vector<std::vector<double>> singleDValuesCol(singledValues.size(), std::vector<double>(1));
        for (int i = 0; i < singledValues.size(); i++)
        {
            singleDValuesCol[i][0] = singledValues[i];
        }

        //Finally we calculate the sample-wise gradient (dInput)
        std::vector<std::vector<double>> gradient = jacobianMatrix * singleDValuesCol;

        // Store it in the dInput
        for(int i=0; i < gradient.size(); i++)
        {
            dInput_[sample][i] = gradient[i][0];
        }
    }

    return dInput_;

}

std::vector<std::vector<double>> NN_Activation::getOutput()
{
    return output_;
}
