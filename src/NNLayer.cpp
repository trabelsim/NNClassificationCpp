#include "NNLayer.h"
#include <iostream>

NN_Layer_Dense::NN_Layer_Dense(int numOfInputFeatures, int numOfNeurons)
{
    createWeightsMatrix(numOfInputFeatures, numOfNeurons);
    createBiasVector(numOfNeurons);
    createWeightsMomentum();
    createBiasMomentum();
}

std::vector<std::vector<double>> NN_Layer_Dense::forward(std::vector<std::vector<double>> & input)
{
    input_ = input;
    auto output = (input_ * weights_) + bias_;
    output_ = output;
    return output_;
}

std::vector<std::vector<double>> NN_Layer_Dense::backward(std::vector<std::vector<double>> &dValues)
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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (int i = 0; i < numOfInputFeatures; i++)
    {
        for (int j = 0; j < numOfNeurons; j++)
        {
            weights_[i][j] = distribution(gen);
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

void NN_Layer_Dense::createWeightsMomentum()
{
    weightsMomentum.resize(weights_.size(), std::vector<double>(weights_[0].size()));
    for(auto &&row : weightsMomentum)
    {
        for(auto &&el : row)
        {
            el = 0.0;
        }
    }
}

void NN_Layer_Dense::createBiasMomentum()
{
    biasMomentum.resize(bias_.size());
    for (auto &&el : biasMomentum)
    {
        el = 0.0;
    }
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

std::vector<std::vector<double>> &NN_Layer_Dense::getWeightsMomentum()
{
    return weightsMomentum;
}

std::vector<double> &NN_Layer_Dense::getBiasMomentum()
{
    return biasMomentum;
}

std::vector<std::vector<double>> NN_Layer_Dense::setWeights(std::vector<std::vector<double>> &newWeights)
{
    weights_.resize(newWeights.size(), std::vector<double>(newWeights[0].size()));
    weights_ = newWeights;
    return weights_;
}

std::vector<double> NN_Layer_Dense::setBias(std::vector<double> &newBias)
{
    bias_.resize(newBias.size());
    bias_ = newBias;
    return bias_;
}

std::vector<std::vector<double>> NN_Layer_Dense::setWeightsMomentum(std::vector<std::vector<double>> &newMomentum)
{
    weightsMomentum.resize(newMomentum.size(), std::vector<double>(newMomentum[0].size()));
    weightsMomentum = newMomentum;
    return weightsMomentum;
}

std::vector<double> NN_Layer_Dense::setBiasMomentum(std::vector<double> &newMomentum)
{
    biasMomentum.resize(newMomentum.size());
    biasMomentum = newMomentum;
    return biasMomentum;
}
