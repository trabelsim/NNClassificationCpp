#include <iostream>
#include "NNLayer.h"
#include "NNInput.h"
#include "NNHelper.h"
#include "NNActivation.h"
#include "NNLoss.h"
#include "NNAccuracy.h"
#include "Optimizers/SGD.h"

using namespace std;

void onlyForward()
{
    cout << "NN Generator started. Have fun!" << endl;

    // Prepare data

    NN_Input nnInput;
    nnInput.spiral_data(100, 3);
    auto spiralInput = nnInput.getInput();

    std::cout << "Input data: " << nnInput.getNumPoints() << " points, " << nnInput.getNumClasses() << " classes." << std::endl;

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

    // Accuracy
    NN_Accuracy accuracyStatistics;
    auto originalSoftmaxMatrix = activationSoftmax.getOutput();
    auto accuracyValue = accuracyStatistics.calculateAccuracy(originalSoftmaxMatrix, groundTruthInput);
    accuracyStatistics.printAccuracy();
}

void forwardAndBackward()
{
    cout << "NN Generator started. Have fun!" << endl;

    // Prepare data

    NN_Input nnInput;
    std::vector<int> groundTruthInput {};
    nnInput.spiral_data(100, 3, groundTruthInput);
    auto spiralInput = nnInput.getInput();

    std::cout << "Input data: " << nnInput.getNumPoints() << " points, " << nnInput.getNumClasses() << " classes." << std::endl;

    // NN Structure
    NN_Layer_Dense layer1(2, 64);
    NN_ActivationReLU activationReLu;
    NN_Layer_Dense layer2(64, 3);
    // SoftMax and classifier combined loss and activation - used to speed up the calculation of the gradients
    NN_ActivationSMaxCategoricalCrossEntropyLoss activationLoss;
    NN_Accuracy accuracyStatistics;
    SGD sgdOptimizer;

    int epoch = 0;
    while(epoch <= 3000)
    {

        // NN Flow - forward
        // std::cout << "FORWARD" << std::endl;
        std::cout << "Epoch: " << epoch;
        auto l1Output = layer1.forward(spiralInput);                    // First layer forward
        auto reLUOutput = activationReLu.forward(l1Output);             // ReLU activation function forward on the hidden layer
        auto l2Output = layer2.forward(reLUOutput);                     // Second layer forward
        auto loss = activationLoss.forward(l2Output, groundTruthInput); // SoftMax and classifier combined loss and activation. - forward with the layer2 output.

        auto lossOutput = activationLoss.getOutput();
        auto accuracy = accuracyStatistics.calculateAccuracy(lossOutput, groundTruthInput);

        std::cout << " loss: " << loss[0] << " accuracy: " << accuracy << std::endl;

        // NN Flow - backward
        // std::cout << "BACKWARD" << std::endl;
        auto dInputLossActivation = activationLoss.backward(lossOutput, groundTruthInput);
        auto dInputLayer2 = layer2.backward(dInputLossActivation);
        auto dInputActivationReLU = activationReLu.backward(dInputLayer2);
        auto dInputLayer1 = layer1.backward(dInputActivationReLU);
        // std::cout << "END" << std::endl;

        sgdOptimizer.updateParameters(layer1);
        sgdOptimizer.updateParameters(layer2);
        epoch++;
    }

    

    // //Some printing - test
    // std::cout << "Layer1" << std::endl;
    // std::cout << "dWeights:" << std::endl;
    // printMatrix(layer1.getdWeights());
    // std::cout << "dBiases:" << std::endl;
    // printVector(layer1.getdBias());

    // std::cout << "Layer2" << std::endl;
    // std::cout << "dWeights:" << std::endl;
    // printMatrix(layer2.getdWeights());
    // std::cout << "dBiases:" << std::endl;
    // printVector(layer2.getdBias());


}


int main()
{
    forwardAndBackward();
    return 0;
}