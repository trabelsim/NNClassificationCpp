#include "NNHelper.h"

class NN_Activation
{
public:
    NN_Activation(){};
    std::vector<std::vector<double>> getOutput();

protected:
    std::vector<std::vector<double>> outputValue;
};

class NN_ActivationReLU : NN_Activation
{
public:
    NN_ActivationReLU(){};
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &matrix);
};

class NN_ActivationSoftMax : public NN_Activation
{
public:
    NN_ActivationSoftMax(){};
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &matrix);
};