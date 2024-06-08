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

std::vector<std::vector<double>> NN_CategoricalCrossEntropyLoss::backward(std::vector<std::vector<double>> &dValues, std::vector<int> &trueValues)
{
    int numOfSamples = dValues.size();
    int numOfLabels = dValues[0].size();
    std::vector<std::vector<double>> yTrueOneHot;

    if(trueValues.size() == 1) // if labels are sparse, we turn them into one-hot enc.
    {
        yTrueOneHot = sLToOneHotEncodedL(trueValues, numOfLabels);
    }

    for(auto &&row : yTrueOneHot)
    {
        for(auto el : row)
        {
            el *= -1.0;
        }
    }

    // Calculate gradient and normalize it.
    dInput_ = (yTrueOneHot / dValues) / numOfSamples;

    return dInput_;
}

std::vector<double> NN_ActivationSMaxCategoricalCrossEntropyLoss::forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{

    std::vector<std::vector<double>> output = nnActivationSoftMax.forward(predictedValues);
    output_ = output;

    auto vecLoss = nnLossCategCrossEntropy.calculate(output, trueValues);

    std::vector<double> vecOutput;
    vecOutput.push_back(vecLoss);

    return vecOutput;
}

std::vector<std::vector<double>> NN_ActivationSMaxCategoricalCrossEntropyLoss::backward(std::vector<std::vector<double>> &dValues, std::vector<int> &trueValues)
{
    int numOfSamples = dValues.size();
    
    dInput_ = dValues;
    
    for(int i=0; i < numOfSamples; i++)
    {
        int y = trueValues[i];
        dInput_[i][y] -= 1.0;
    }

    return dInput_;
}

std::vector<std::vector<double>> NN_ActivationSMaxCategoricalCrossEntropyLoss::getOutput()
{
    return output_;
}
