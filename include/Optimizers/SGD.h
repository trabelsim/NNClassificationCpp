#include "BaseOptimizer.h"
#include "NNLayer.h"

class SGD : public BaseOptimizer
{
public:
    SGD(const double& learningRate = 1.0) : BaseOptimizer(learningRate) {};
    void updateParameters(NN_Layer_Dense& parametersLayer);
};