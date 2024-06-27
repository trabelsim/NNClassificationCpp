#include "NNHelper.h"

class BaseOptimizer
{
public:
    BaseOptimizer();

protected:
    double learningRate_ {0.8};
};