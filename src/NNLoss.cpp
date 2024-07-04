#include "NNLoss.h"
#include <iostream>

double NN_Loss::calculate(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    auto vecOutput = this->forward(predictedValues, trueValues);
    // std::cout << "Loss calculation vector " << std::endl;
    // printVector(vecOutput);
    auto n = vecOutput.size();
    // std::cout << "Size of vec " << n << std::endl;


    // std::cout << "true values" << std::endl;
    // printVector(trueValues);

    lossValue = 0;

    for (int i = 0; i < n; i++)
    {
        lossValue += vecOutput[i];
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
    int n = predictedValues.size();
    std::vector<double> losses(n);

    for (int i = 0; i < n; ++i)
    {
        int trueLabel = trueValues[i];
        losses[i] = -std::log(std::max(predictedValues[i][trueLabel], 1e-15)); // Prevent log(0)
    }

    return losses;
}

std::vector<std::vector<double>> NN_CategoricalCrossEntropyLoss::backward(std::vector<std::vector<double>> &dValues, std::vector<int> &trueValues)
{
    int numOfSamples = dValues.size();
    int numOfLabels = dValues[0].size();
    std::vector<std::vector<double>> yTrueOneHot;

    if(trueValues.size() == 1) // if labels are sparse, we turn them into one-hot enc.
    {
        yTrueOneHot = sLToOneHotEncodedL(trueValues, numOfLabels);
    }

   // If labels are sparse, modify the gradient based on one-hot encoding
    // if (!yTrueOneHot.empty())
    // {
    for (int i = 0; i < numOfSamples; i++)
    {
        for (int j = 0; j < numOfLabels; j++)
        {
            dInput_[i][j] /= -yTrueOneHot[i][j];
        }
    }
    // }

    // Normalize gradient
    for (auto &&row : dInput_)
    {
        for (auto &&value : row)
        {
            value /= numOfSamples;
        }
    }

    return dInput_;
}
