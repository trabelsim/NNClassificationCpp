#include "NNHelper.h"

class NN_Accuracy
{
public:
    NN_Accuracy(){};
    double calculateAccuracy(std::vector<std::vector<double>> &predictedValues, std::vector<int> &trueValues);
    double getAccuracy();
    void printAccuracy();

private:
    double accuracy = 0;
};