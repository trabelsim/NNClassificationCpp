#include <vector>
#include "NNHelper.h"

class NN_Layer_Dense
{
    public:
        NN_Layer_Dense(int numOfInputFeatures, int numOfNeurons); // Steps to perform: 1)know the size of the input and the num of neurons we want to create, 2) normalize the data,

        std::vector<std::vector<float>> forward(std::vector<std::vector<float>> &input);

        /*
            Getters
        */
        std::vector<std::vector<float>> getWeights();
        std::vector<float> getBias();
        std::vector<std::vector<float>> getOutput();

    private:
        std::vector<std::vector<float>> weights_;
        std::vector<float> bias_;
        std::vector<std::vector<float>> output_;

        std::vector<std::vector<float>> createWeightsMatrix(int &numOfInputFeatures, int &numOfNeurons);
        std::vector<float> createBiasVector(int &numOfNeurons);
};