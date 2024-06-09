#include "Optimizers/SGD.h"
#include "NNHelper.h"
#include <iostream>

void SGD::updateParameters(NN_Layer_Dense &parametersLayer)
{
    // std::cout << "OLD WEIGHTS" << std::endl;
    // printMatrix(parametersLayer.getWeights());
    auto newWeights = -learningRate_ * parametersLayer.getdWeights();
    newWeights = newWeights + parametersLayer.getWeights();
    parametersLayer.setWeights(newWeights);
    // std::cout << "NEW WEIGHTS" << std::endl;
    // printMatrix(parametersLayer.getWeights());

    // std::cout << "OLD BIASES" << std::endl;
    // printVector(parametersLayer.getBias());
    auto newBiases = -learningRate_ * parametersLayer.getdBias();
    newBiases = newBiases + parametersLayer.getBias();
    parametersLayer.setBias(newBiases);
    // std::cout << "NEW BIASES" << std::endl;
    // printVector(parametersLayer.getBias());
}