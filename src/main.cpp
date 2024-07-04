#include <iostream>
#include "NNLayer.h"
#include "NNInput.h"
#include "NNHelper.h"
#include "NNActivation.h"
#include "NNLoss.h"
#include "NNAccuracy.h"
#include "NNActivationLoss.h"
#include "Optimizers/SGD.h"
#include <unistd.h>

using namespace std;

void printAsciiArt()
{
    const std::string artText = R"(

  ____                   _   _                      _ 
 / ___| _     _         | \ | | ___ _   _ _ __ __ _| |
| |   _| |_ _| |_       |  \| |/ _ \ | | | '__/ _` | |
| |__|_   _|_   _|      | |\  |  __/ |_| | | | (_| | |
 \____||_|  _|_|        |_| \_|\___|\__,_|_|  \__,_|_|
 _ __   ___| |___      _____  _ __| | __         
| '_ \ / _ \ __\ \ /\ / / _ \| '__| |/ /    o-----o-----o     
| | | |  __/ |_ \ V  V / (_) | |  |   <    / \   / \   / \      
|_| |_|\___|\__| \_/\_/ \___/|_|  |_|\_\  o   o-o   o-o   o       
                                           \ /   \ /   \ /
                                            o-----o-----o
)";

    std::cout << artText << std::endl;
}



void printInputDataHeader(int numPoints, int numClasses, int numNeurons)
{
    std::cout << std::setw(25) << std::left << "Points"
                << std::setw(20) << std::left << "Classes"
                << std::setw(20) << std::left << "Neurons" << std::endl;
    std::cout << std::string(53, '-') << std::endl; // Draw a line for table separation

    std::cout << std::setw(25) << std::left << numPoints
                << std::setw(20) << std::left << numClasses
                << std::setw(20) << std::left << numNeurons << std::endl;
    std::cout << std::string(53, '-') << std::endl; // Line for table separation
}

void printHyperparametersTableHeader(double learningRateDecay, double momentum, double initialLearningRate, int numEpochsMax)
{
    std::cout << std::setw(16) << std::left << "Learning Rate"
                << std::setw(17) << std::left << "Decay"
                << std::setw(12) << std::left << "Momentum"
                << std::setw(13) << std::left << "Epochs" << std::endl;
    std::cout << std::string(53, '-') << std::endl; // Draw a line for table separation

    std::cout << std::setw(16) << std::left << std::fixed << std::setprecision(6) << initialLearningRate
                << std::setw(17) << std::left << std::fixed << std::setprecision(6) << learningRateDecay
                << std::setw(12) << std::left << std::fixed << std::setprecision(6) << momentum
                << std::setw(13) << std::left << numEpochsMax << std::endl;
    std::cout << std::string(53, '-') << std::endl; // Line for table separation
}

void printTableHeader()
{
    std::cout << std::string(53, '-') << std::endl; // Line for table separation
    std::cout << std::setw(8) << std::left << "Epoch"
                << std::setw(15) << std::left << "Loss"
                << std::setw(15) << std::left << "Accuracy"
                << std::setw(15) << std::left << "Learning Rate" << std::endl;
    std::cout << std::string(53, '-') << std::endl; // Line for table separation
}

void printRecord(int epoch, double loss, double accuracy, double learningRate)
{
    std::cout << std::string(53, '-') << std::endl; // Line for table separation
    std::cout << std::setw(8) << std::left << epoch
                << std::setw(15) << std::left << std::fixed << std::setprecision(6) << loss
                << std::setw(15) << std::left << std::fixed << std::setprecision(6) << accuracy
                << std::setw(15) << std::left << std::fixed << std::setprecision(6) << learningRate << std::endl;
    std::cout << std::string(53, '-') << std::endl; // Line for table separation
}

static constexpr int NUM_OF_POINTS = 300;
static constexpr int NUM_OF_CLASSES = 3;
static constexpr int NUM_OF_NEURONS = 64;

static constexpr double LEARNING_RATE_DECAY = 1e-4;
static constexpr double LEARNING_MOMENTUM = 0.5;

static constexpr int EPOCH_DIVIDER = 100;
static constexpr int EPOCH_DATA_VISUAL_LIMIT = 7;

void forwardAndBackward(int numOfEpochs, double initialLearningRate)
{
    printAsciiArt();
    printInputDataHeader(NUM_OF_POINTS, NUM_OF_CLASSES, NUM_OF_NEURONS);
    printHyperparametersTableHeader(LEARNING_RATE_DECAY, LEARNING_MOMENTUM, initialLearningRate, numOfEpochs);
    sleep(3);
    // Prepare data
    NN_Input nnInput;
    std::vector<int> groundTruthInput{};
    std::vector<std::vector<double>> spiralInput = nnInput.spiral_data(NUM_OF_POINTS, NUM_OF_CLASSES, groundTruthInput);

    // NN Structure
    NN_Layer_Dense layer1(2, 64);
    NN_ActivationReLU activationReLu;
    NN_Layer_Dense layer2(64, 3);
    NN_ActivationSMaxCategoricalCrossEntropyLoss activationLoss;
    NN_ActivationSoftMax activationSoftMax;
    NN_CategoricalCrossEntropyLoss categCrossEntropyloss;
    NN_Accuracy accuracyStatistics;
    SGD sgdOptimizer(LEARNING_MOMENTUM, initialLearningRate);
    double learningRate = initialLearningRate;

    int epoch = 0;
    printTableHeader();
    while (epoch < numOfEpochs)
    {

        // NN Flow - forward
        auto l1Output = layer1.forward(spiralInput);                    // First layer forward
        auto reLUOutput = activationReLu.forward(l1Output);             // ReLU activation function forward on the hidden layer
        auto l2Output = layer2.forward(reLUOutput);                     // Second layer forward
        auto loss = activationLoss.forward(l2Output, groundTruthInput); // SoftMax and classifier combined loss and activation. - forward with the layer2 output.

        auto lossOutput = activationLoss.getOutput();
        auto accuracy = accuracyStatistics.calculateAccuracy(lossOutput, groundTruthInput);

        // NN Flow - backward
        auto dInputLossActivation = activationLoss.backward(lossOutput, groundTruthInput);
        auto dInputLayer2 = layer2.backward(dInputLossActivation);
        auto dInputActivationReLU = activationReLu.backward(dInputLayer2);
        auto dInputLayer1 = layer1.backward(dInputActivationReLU);

        sgdOptimizer.updateLearningRate(LEARNING_RATE_DECAY, learningRate);
        sgdOptimizer.updateParameters(layer1, learningRate);
        sgdOptimizer.updateParameters(layer2, learningRate);

        if(epoch % EPOCH_DIVIDER == 0)
        {
            if (epoch % (EPOCH_DIVIDER * EPOCH_DATA_VISUAL_LIMIT) == 0 && (epoch != 0))
            {
                printTableHeader();
            }

            printRecord(epoch, loss[0], accuracy, learningRate);
        }
        
        epoch++;
    }
}

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    forwardAndBackward(10000, 1.0);
    return 0;
}