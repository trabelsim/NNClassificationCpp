#include "NNAccuracy.h"
#include <iostream>

double NN_Accuracy::calculateAccuracy(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    int n = predictedValues.size();
    std::vector<int> predictions(n);

    for (int i = 0; i < n; ++i)
    {
        double max_val = predictedValues[i][0];
        int max_index = 0;
        for (int j = 1; j < predictedValues[i].size(); ++j)
        {
            if (predictedValues[i][j] > max_val)
            {
                max_val = predictedValues[i][j];
                max_index = j;
            }
        }
        predictions[i] = max_index;
    }

    int correct_predictions = 0;
    for (int i = 0; i < n; ++i)
    {
        if (predictions[i] == trueValues[i])
        {
            correct_predictions++;
        }
    }

    accuracy = static_cast<double>(correct_predictions) / n;
    return accuracy;
}

double NN_Accuracy::getAccuracy()
{
    return accuracy;
}

void NN_Accuracy::printAccuracy()
{
    std::cout << "Accuracy: " << std::setprecision(5) << accuracy << std::endl;
}
