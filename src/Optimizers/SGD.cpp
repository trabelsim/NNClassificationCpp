#include "Optimizers/SGD.h"
#include "NNHelper.h"
#include <iostream>

void SGD::updateParameters(NN_Layer_Dense &parametersLayer, double learningRate)
{
    auto newWeights = -learningRate * parametersLayer.getdWeights();
    newWeights = parametersLayer.getWeights() + newWeights;
    parametersLayer.setWeights(newWeights);

    auto newBiases = -learningRate * parametersLayer.getdBias();
    newBiases = parametersLayer.getBias() + newBiases;
    parametersLayer.setBias(newBiases);
}