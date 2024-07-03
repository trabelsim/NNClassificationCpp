#include "NNHelper.h"
#include "NNActivation.h"
#include "NNLoss.h"

class NN_ActivationSMaxCategoricalCrossEntropyLoss
{
public:
    NN_ActivationSMaxCategoricalCrossEntropyLoss(){};
    std::vector<double> forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues);
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &dValues, std::vector<int> &trueValues);
    std::vector<std::vector<double>> &getOutput();

private:
    NN_ActivationSoftMax nnActivationSoftMax;
    NN_CategoricalCrossEntropyLoss nnLossCategCrossEntropy;
    std::vector<std::vector<double>> output_;
};