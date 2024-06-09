#ifndef NN_LOSS_H
#define NN_LOSS_H

#include "NNHelper.h"
#include "NNActivation.h"

class NN_Loss
{
public:
    NN_Loss(){};
    double calculate(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues);
    double getLoss();
    void printLoss();
protected:
    virtual std::vector<double> forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues){};
    double lossValue = 0;

    virtual std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &dValues, std::vector<int> &trueValues) {};
    std::vector<std::vector<double>> dInput_;
};


class NN_CategoricalCrossEntropyLoss : public NN_Loss
{
public:
    NN_CategoricalCrossEntropyLoss(){};
    std::vector<double> forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues) override;
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &dValues, std::vector<int> &trueValues) override;
};

class NN_ActivationSMaxCategoricalCrossEntropyLoss : public NN_Loss
{
public:
    NN_ActivationSMaxCategoricalCrossEntropyLoss(){};
    std::vector<double> forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues) override;
    std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &dValues, std::vector<int> &trueValues) override;
    std::vector<std::vector<double>> getOutput();

private:
    NN_ActivationSoftMax nnActivationSoftMax;
    NN_CategoricalCrossEntropyLoss nnLossCategCrossEntropy;
    std::vector<std::vector<double>> output_;
};

#endif NN_LOSS_H