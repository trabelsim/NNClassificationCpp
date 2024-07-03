#include "NNLayer.h"
#include <iostream>

NN_Layer_Dense::NN_Layer_Dense(int numOfInputFeatures, int numOfNeurons)
{
    createWeightsMatrix(numOfInputFeatures, numOfNeurons);
    createBiasVector(numOfNeurons);
}

std::vector<std::vector<double>> &NN_Layer_Dense::forward(std::vector<std::vector<double>> & input)
{
    input_ = input;
    auto output = (input_ * weights_) + bias_;
    output_ = output;
    return output_;
}

std::vector<std::vector<double>>& NN_Layer_Dense::backward(std::vector<std::vector<double>> &dValues)
{
    auto transposedInput = transpose(input_);
    dWeights_ = transposedInput * dValues;
    dBiases_ = sumElementsOnAxisZero(dValues);
    auto transposedWeights = transpose(weights_);
    dInputs_ = dValues * transposedWeights;
    return dInputs_;
}

std::vector<std::vector<double>> NN_Layer_Dense::createWeightsMatrix(int &numOfInputFeatures, int &numOfNeurons)
{
    weights_.resize(numOfInputFeatures,std::vector<double>(numOfNeurons));
    double std_dev = sqrt(2.0 / (numOfInputFeatures + numOfNeurons)); //Xavier init.
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, std_dev);
    for (int i = 0; i < numOfInputFeatures; i++)
    {
        for (int j = 0; j < numOfNeurons; j++)
        {
            weights_[i][j] = distribution(generator);
        }
    }

    return weights_;
}

std::vector<double> NN_Layer_Dense::createBiasVector(int &numOfNeurons)
{
    bias_.resize(numOfNeurons);
    bias_ = std::vector<double>(numOfNeurons, 0.0);
    return bias_;
}

std::vector<std::vector<double>>& NN_Layer_Dense::getWeights()
{
    return weights_;
}

std::vector<double>& NN_Layer_Dense::getBias()
{
    return bias_;
}

std::vector<std::vector<double>>& NN_Layer_Dense::getOutput()
{
    return output_;
}

std::vector<std::vector<double>>& NN_Layer_Dense::getdWeights()
{
    return dWeights_;
}

std::vector<double>& NN_Layer_Dense::getdBias()
{
    return dBiases_;
}

std::vector<std::vector<double>>& NN_Layer_Dense::getdInput()
{
    return dInputs_;
}

std::vector<std::vector<double>> NN_Layer_Dense::setWeights(std::vector<std::vector<double>> &newWeights)
{
    weights_ = newWeights;
    return weights_;
}

std::vector<double> NN_Layer_Dense::setBias(std::vector<double> &newBias)
{
    bias_ = newBias;
    return bias_;
}
