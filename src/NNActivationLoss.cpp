#include "NNActivationLoss.h"

std::vector<double> NN_ActivationSMaxCategoricalCrossEntropyLoss::forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{

    std::vector<std::vector<double>> output = nnActivationSoftMax.forward(predictedValues);
    output_ = output;

    auto vecLoss = nnLossCategCrossEntropy.calculate(output_, trueValues);

    std::vector<double> vecOutput;
    vecOutput.push_back(vecLoss);

    return vecOutput;
}

// std::vector<std::vector<double>> NN_ActivationSMaxCategoricalCrossEntropyLoss::backward(std::vector<std::vector<double>> &dValues, std::vector<int> &trueValues)
// {
//     int numOfSamples = dValues.size();
//     int numOfClasses = dValues[0].size();
//     std::vector<std::vector<double>> dINputs(numOfSamples, std::vector<double>(numOfClasses));

//     for (int i = 0; i < numOfSamples; ++i)
//     {
//         dINputs[i] = dValues[i];
//         dINputs[i][trueValues[i]] -= 1.0;
//         dINputs[i] = dINputs[i] / numOfSamples;
//     }

//     return dINputs;
// }

std::vector<std::vector<double>> NN_ActivationSMaxCategoricalCrossEntropyLoss::backward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    int n = predictedValues.size();    // Number of samples
    int m = predictedValues[0].size(); // Number of classes
    std::vector<std::vector<double>> dInputs(n, std::vector<double>(m));
    
    for (int i = 0; i < n; ++i)
    {
        dInputs[i] = predictedValues[i];
        dInputs[i][trueValues[i]] -= 1.0; // Gradient calculation
    }

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            dInputs[i][j] /= n; // Normalize gradients by number of samples
        }
    }

    return dInputs;
}


std::vector<std::vector<double>> &NN_ActivationSMaxCategoricalCrossEntropyLoss::getOutput()
{
    return output_;
}