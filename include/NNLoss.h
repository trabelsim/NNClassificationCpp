
#include "NNHelper.h"

class NN_Loss
{
public:
    NN_Loss(){};
    double calculate(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues);
protected:
    virtual std::vector<double> forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues){};
};


class NN_CategoricalCrossEntropyLoss : NN_Loss
{
public:
    NN_CategoricalCrossEntropyLoss(){};
    double calculate(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues);

private:
    std::vector<double> forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues) override;
};