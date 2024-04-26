#include "NNAccuracy.h"
#include <iostream>

double NN_Accuracy::calculateAccuracy(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues)
{
    int n = getNumOfRows(predictedValues);
    int m = getNumOfColumns(predictedValues);
    std::vector<double> predictionVector(n, 0);
    std::vector<double> truthValuesVector(n, 0);

    // get the maximum value and push it to the predictionMatrix
    // get the ground truth index and assing the value from predictedValues to the vector of true values.
    for (int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            if (predictedValues[i][j] > predictionVector[i])
            {
                predictionVector[i] = predictedValues[i][j];
                truthValuesVector[i] = predictedValues[i][trueValues[i]];
            }
        }
    }

    // Get the ground truth values based on the trueValues indexe vector
    // Next compare it to the predictionVector values
    // For a prediction = ground truth -> increase the counter of matched values.
    int matchedValues = 0;
    for (int i = 0; i < n; i++)
    {
        if(truthValuesVector[i] == predictionVector[i])
        {
            matchedValues++;
        }
    }

    accuracy = (double)matchedValues / (double)n;
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
