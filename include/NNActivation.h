#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "NNHelper.h"

class NN_Activation
{
public:
    NN_Activation(){};
    std::vector<std::vector<double>> getOutput();

protected:
    // Store vars for backward.
    std::vector<std::vector<double>> output_;
    std::vector<std::vector<double>> input_;
    std::vector<std::vector<double>> dInput_;
};

class NN_ActivationReLU : NN_Activation
{
public:
    NN_ActivationReLU(){};
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &matrix);
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &matrix);
};

class NN_ActivationSoftMax : public NN_Activation
{
public:
    NN_ActivationSoftMax(){};
    std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &matrix);
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &matrix);
};

#endif // NN_ACTIVATION_H