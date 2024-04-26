#include <iostream>
#include "NNLayer.h"
#include "NNInput.h"
#include "NNHelper.h"
#include "NNActivation.h"
#include "NNLoss.h"
#include "NNAccuracy.h"

using namespace std;

int main()
{
    cout << "NNGen" << endl;

    // Prepare data

    NN_Input nnInput;
    nnInput.spiral_data(100,3);
    auto spiralInput = nnInput.getInput();

    NN_Layer_Dense layer1(2, 3);
    auto l1Output = layer1.forward(spiralInput); // First layer forward

    NN_ActivationReLU activationReLu;
    auto reLUOutput = activationReLu.forward(l1Output); // ReLU activation function forward on the hidden layer

    NN_Layer_Dense layer2(3, 3);
    auto l2Output = layer2.forward(reLUOutput); // Second layer forward

    NN_ActivationSoftMax activationSoftmax;
    auto softMaxOut = activationSoftmax.forward(l2Output); // Softmax activation function forward on the output layer

    // Loss categorical cross entropy
    NN_CategoricalCrossEntropyLoss activationLossCategCrossEntropy;
    auto groundTruthInput = nnInput.getTrueInput();
    activationLossCategCrossEntropy.calculate(softMaxOut, groundTruthInput);
    activationLossCategCrossEntropy.printLoss();

    //Accuracy
    NN_Accuracy accuracyStatistics;
    auto originalSoftmaxMatrix = activationSoftmax.getOutput();
    auto accuracyValue = accuracyStatistics.calculateAccuracy(originalSoftmaxMatrix, groundTruthInput);
    accuracyStatistics.printAccuracy();

    return 0;
}