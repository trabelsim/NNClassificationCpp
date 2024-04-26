#include "NNLoss.h"
#include <iostream>

double NN_Loss::calculate(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    auto matrixOutput = this->forward(predictedValues, trueValues);
    auto n = matrixOutput.size();
    double retLoss = 0;

    for (int i = 0; i < n; i++)
    {
        retLoss += matrixOutput[i];
    }

    retLoss = retLoss / n;

    return retLoss;
}

double NN_CategoricalCrossEntropyLoss::calculate(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    return NN_Loss::calculate(predictedValues, trueValues);
}

std::vector<double> NN_CategoricalCrossEntropyLoss::forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    auto n = getNumOfRows(predictedValues);
    std::vector<double> matrixOutput(n);

    clipValues(predictedValues);

    // Selecting the values based on the ground truth and pushing them to the matrix
    // and calculating the negative log of each element in the matrix.
    for (int i = 0; i < n; i++)
    {
        matrixOutput[i] = -log(predictedValues[i][trueValues[i]]);
    }
    
    return matrixOutput;
}

