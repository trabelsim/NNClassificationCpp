#include "Optimizers/SGD.h"
#include "NNHelper.h"
#include <iostream>
#include "SGD.h"

SGD::SGD(double momentumVal, double baseLearningRateVal)
{
    if(momentumVal >= 0.0)
    {
        momentum = momentumVal;
        momentumActive = true;
    }

    baseLearningRate = baseLearningRateVal;
}

void SGD::updateParameters(NN_Layer_Dense &parametersLayer, double learningRate) 
{
    if(momentumActive)
    {
        //Weights
        auto momentumWeightsUpdate = momentum * parametersLayer.getWeightsMomentum() - learningRate * parametersLayer.getdWeights();
        parametersLayer.setWeightsMomentum(momentumWeightsUpdate);

        auto newWeights = parametersLayer.getWeights() + momentumWeightsUpdate;
        parametersLayer.setWeights(newWeights);

        //Bias
        auto momentumBiasUpdate = momentum * parametersLayer.getBiasMomentum() - learningRate * parametersLayer.getdBias();
        parametersLayer.setBiasMomentum(momentumBiasUpdate);

        auto newBias = parametersLayer.getBias() + momentumBiasUpdate;
        parametersLayer.setBias(newBias);
    }
    else
    {
        auto newWeights = -learningRate * parametersLayer.getdWeights();
        newWeights = parametersLayer.getWeights() + newWeights;
        parametersLayer.setWeights(newWeights);

        auto newBiases = -learningRate * parametersLayer.getdBias();
        newBiases = parametersLayer.getBias() + newBiases;
        parametersLayer.setBias(newBiases);
    }

    iteration++;
}

void SGD::updateLearningRate(double learningRateDecay, double &learningRate)
{
    if(iteration % 10 == 0)
    {
        learningRate = baseLearningRate * exp(-learningRateDecay * iteration);
    }
    // learningRate = baseLearningRate * (1 / (1 + learningRateDecay * iteration));
     //(1 / (1 + learningRateDecay * iteration));
}