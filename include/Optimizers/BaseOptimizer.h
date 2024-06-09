#include "NNHelper.h"

class BaseOptimizer
{
public:
    BaseOptimizer(const double& learningRate) : learningRate_{learningRate} {};

protected:
    double learningRate_ {0.1};
};