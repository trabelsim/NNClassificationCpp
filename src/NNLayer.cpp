#include "NNLayer.h"

NN_Layer_Dense::NN_Layer_Dense(int numOfInputFeatures, int numOfNeurons) : weights_(createWeightsMatrix(numOfInputFeatures, numOfNeurons)),
                                                                           bias_(createBiasVector(numOfNeurons))
{

}

std::vector<std::vector<float>> NN_Layer_Dense::forward(std::vector<std::vector<float>> &input)
{
    auto output = (input * weights_) + bias_;
    output_ = output;
    return output_;
}

std::vector<std::vector<float>> NN_Layer_Dense::createWeightsMatrix(int &numOfInputFeatures, int &numOfNeurons)
{
    std::vector<std::vector<float>> weights(numOfInputFeatures, std::vector<float>(numOfNeurons));
    for (int i = 0; i < numOfInputFeatures; i++)
    {
        for (int j = 0; j < numOfNeurons; j++)
        {
            weights[i][j] = randomGenerator(-1, 1);
        }
    }

    return weights;
}

std::vector<float> NN_Layer_Dense::createBiasVector(int &numOfNeurons)
{
    std::vector<float> biasVector(numOfNeurons);
    for (int i = 0; i < numOfNeurons; i++)
    {
        biasVector[i] = 0;
    }

    return biasVector;
}

std::vector<std::vector<float>> NN_Layer_Dense::getWeights()
{
    return weights_;
}

std::vector<float> NN_Layer_Dense::getBias()
{
    return bias_;
}

std::vector<std::vector<float>> NN_Layer_Dense::getOutput()
{
    return output_;
}