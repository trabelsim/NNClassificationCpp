#include <iostream>
#include "NNLayer.h"
#include "NNInput.h"
#include "NNHelper.h"
#include "NNActivation.h"
#include "NNLoss.h"
#include "NNAccuracy.h"
#include "NNActivationLoss.h"
#include "Optimizers/SGD.h"

using namespace std;

void forwardAndBackward(int numOfEpochs, double initialLearningRate)
{
    cout << "NN trainer" << endl;


    double learningRateDecay = 1e-3;
    // Prepare data
    NN_Input nnInput;
    std::vector<int> groundTruthInput {};
    std::vector<std::vector<double>> spiralInput = nnInput.spiral_data(100, 3, groundTruthInput);

    std::cout << "Input data: " << nnInput.getNumPoints() << " points, " << nnInput.getNumClasses() << " classes." << std::endl;

    // NN Structure
    NN_Layer_Dense layer1(2, 64);
    NN_ActivationReLU activationReLu;
    NN_Layer_Dense layer2(64, 3);
    // SoftMax and classifier combined loss and activation - used to speed up the calculation of the gradients
    NN_ActivationSMaxCategoricalCrossEntropyLoss activationLoss; // kuku
    NN_ActivationSoftMax activationSoftMax;
    NN_CategoricalCrossEntropyLoss categCrossEntropyloss;
    NN_Accuracy accuracyStatistics;
    SGD sgdOptimizer;
    double learningRate = initialLearningRate;

    int epoch = 0;
    while (epoch < numOfEpochs)
    {

        // NN Flow - forward
        std::cout << "Epoch: " << epoch;
        auto l1Output = layer1.forward(spiralInput);                    // First layer forward
        auto reLUOutput = activationReLu.forward(l1Output);             // ReLU activation function forward on the hidden layer
        auto l2Output = layer2.forward(reLUOutput);                     // Second layer forward
        auto loss = activationLoss.forward(l2Output, groundTruthInput); // SoftMax and classifier combined loss and activation. - forward with the layer2 output.
        
        auto lossOutput = activationLoss.getOutput();
        auto accuracy = accuracyStatistics.calculateAccuracy(lossOutput, groundTruthInput);
        std::cout << " loss: " << loss[0] << " accuracy: " << accuracy << std::endl;

        // NN Flow - backward
        auto dInputLossActivation = activationLoss.backward(lossOutput, groundTruthInput);
        auto dInputLayer2 = layer2.backward(dInputLossActivation);
        auto dInputActivationReLU = activationReLu.backward(dInputLayer2);
        auto dInputLayer1 = layer1.backward(dInputActivationReLU);

        sgdOptimizer.updateParameters(layer1, learningRate);
        sgdOptimizer.updateParameters(layer2, learningRate);

        epoch++;

        if (epoch % 100 == 0)
        {
            // learning rate decay

            learningRate /= (1/(1+ learningRateDecay * epoch)); 
        }

        // double max_gradient_magnitude = 0.0;
        // auto dweightslayer1 = layer1.getdWeights();
        // for (int i = 0; i < layer1.getWeights().size(); ++i)
        // {
        //     for (int j = 0; j < layer1.getWeights()[0].size(); j++)
        //     {
        //         double gradient_magnitude = std::abs(dweightslayer1[i][j]);
        //         if (gradient_magnitude > max_gradient_magnitude)
        //         {
        //             max_gradient_magnitude = gradient_magnitude;
        //         }
        //     }
        // }

        // Learning rate decay every 100 epochs
    }
}


int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    forwardAndBackward(5000, 1);
    return 0;
}