#include "NNLayer.h"

NN_Layer_Dense::NN_Layer_Dense(int numOfInputFeatures, int numOfNeurons) : weights_(createWeightsMatrix(numOfInputFeatures, numOfNeurons)),
                                                                           bias_(createBiasVector(numOfNeurons))
{

}

std::vector<std::vector<double>> NN_Layer_Dense::forward(std::vector<std::vector<double>> &input)
{
    input_ = input;
    auto output = (input * weights_) + bias_;
    output_ = output;
    return output_;
}

std::vector<std::vector<double>> NN_Layer_Dense::backward(std::vector<std::vector<double>> &dValues)
{
    dWeights_ = transpose(input_) * dValues;
    dBiases_ = sumElementsOnAxisZero(dValues);
    dInputs_ = dValues * transpose(weights_);
    return dInputs_;
}

std::vector<std::vector<double>> NN_Layer_Dense::createWeightsMatrix(int &numOfInputFeatures, int &numOfNeurons)
{
    std::vector<std::vector<double>> weights(numOfInputFeatures, std::vector<double>(numOfNeurons));
    for (int i = 0; i < numOfInputFeatures; i++)
    {
        for (int j = 0; j < numOfNeurons; j++)
        {
            weights[i][j] = WEIGHTS_NORMALIZER * randomGenerator(0, 1);
        }
    }

    return weights;
}

std::vector<double> NN_Layer_Dense::createBiasVector(int &numOfNeurons)
{
    std::vector<double> biasVector(numOfNeurons);
    for (int i = 0; i < numOfNeurons; i++)
    {
        biasVector[i] = 0;
    }

    return biasVector;
}

std::vector<std::vector<double>> NN_Layer_Dense::getWeights()
{
    return weights_;
}

std::vector<double> NN_Layer_Dense::getBias()
{
    return bias_;
}

std::vector<std::vector<double>> NN_Layer_Dense::getOutput()
{
    return output_;
}

std::vector<std::vector<double>> NN_Layer_Dense::getdWeights()
{
    return dWeights_;
}

std::vector<std::vector<double>> NN_Layer_Dense::getdBias()
{
    return dBiases_;
}

std::vector<std::vector<double>> NN_Layer_Dense::getdInput()
{
    return dInputs_;
}
