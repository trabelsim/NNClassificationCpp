
#include "NNHelper.h"

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
};


class NN_CategoricalCrossEntropyLoss : public NN_Loss
{
public:
    NN_CategoricalCrossEntropyLoss(){};

private:
    std::vector<double> forward(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues) override;
};