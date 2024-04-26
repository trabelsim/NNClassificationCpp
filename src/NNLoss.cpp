#include "NNLoss.h"
#include <iostream>

double NN_Loss::calculate(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    auto matrixOutput = this->forward(predictedValues, trueValues);
    auto n = matrixOutput.size();

    lossValue = 0;

    for (int i = 0; i < n; i++)
    {
        lossValue += matrixOutput[i];
    }

    lossValue = lossValue / n;

    return lossValue;
}

double NN_Loss::getLoss()
{
    return lossValue;
}

void NN_Loss::printLoss()
{
    std::cout << "Loss: " << lossValue << std::endl;
}


std::vector<double> NN_CategoricalCrossEntropyLoss::forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    auto n = getNumOfRows(predictedValues);
    std::vector<double> matrixOutput(n);

    auto clippedValues = clipValues(predictedValues);

    // Selecting the values based on the ground truth and pushing them to the matrix
    // and calculating the negative log of each element in the matrix.
    for (int i = 0; i < n; i++)
    {
        matrixOutput[i] = -log(clippedValues[i][trueValues[i]]);
    }
    
    return matrixOutput;
}

