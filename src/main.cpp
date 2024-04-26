#include <iostream>
#include "NNLayer.h"
#include "NNInput.h"
#include "NNHelper.h"
#include "NNActivation.h"
#include "NNLoss.h"

using namespace std;

int main()
{
    cout << "NNGen" << endl;

    // Prepare data

    auto nnInput = NN_Input();
    nnInput.spiral_data(100,3);
    auto spiralInput = nnInput.getInput();
    

    auto layer1 = NN_Layer_Dense(2, 3);
    auto l1Output = layer1.forward(spiralInput); // First layer forward

    auto activationReLu = NN_ActivationReLU();
    auto reLUOutput = activationReLu.forward(l1Output); // ReLU activation function forward on the hidden layer

    auto layer2 = NN_Layer_Dense(3, 3);
    auto l2Output = layer2.forward(reLUOutput); // Second layer forward

    auto activationSoftmax = NN_ActivationSoftMax();
    auto softMaxOut = activationSoftmax.forward(l2Output); // Softmax activation function forward on the output layer

    printMatrix(softMaxOut);

    // Loss categorical cross entropy
    auto activationLossCategCrossEntropy = NN_CategoricalCrossEntropyLoss();
    auto groundTruthInput = nnInput.getTrueInput();
    auto loss = activationLossCategCrossEntropy.calculate(softMaxOut, groundTruthInput);

    std::cout << "Loss: " << std::setprecision(10) <<loss << std::endl;

    return 0;
}