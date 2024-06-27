#include "BaseOptimizer.h"
#include "NNLayer.h"

class SGD
{
public:
    SGD() = default;
    void updateParameters(NN_Layer_Dense &parametersLayer, double learningRate);
};