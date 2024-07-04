#ifndef NN_SGD_H
#define NN_SGD_H

#include "BaseOptimizer.h"
#include "NNLayer.h"

class SGD
{
public:
    SGD(double momentumVal, double baseLearningRateVal);
    void updateLearningRate(double learningRateDecay, double& learningRate);
    void updateParameters(NN_Layer_Dense &parametersLayer, double learningRate);

private:
    int iteration {0};
    bool momentumActive {false};
    double momentum {0.0};
    double baseLearningRate {0.0};
};

#endif //NN_SGD_H