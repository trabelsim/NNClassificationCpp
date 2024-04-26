#include <vector>
#include "NNHelper.h"

class NN_Layer_Dense
{
    public:
        NN_Layer_Dense(int numOfInputFeatures, int numOfNeurons); // Steps to perform: 1)know the size of the input and the num of neurons we want to create, 2) normalize the data,

        std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &input);

        /*
            Getters
        */
        std::vector<std::vector<double>> getWeights();
        std::vector<double> getBias();
        std::vector<std::vector<double>> getOutput();

    private:
        std::vector<std::vector<double>> weights_;
        std::vector<double> bias_;
        std::vector<std::vector<double>> output_;

        std::vector<std::vector<double>> createWeightsMatrix(int &numOfInputFeatures, int &numOfNeurons);
        std::vector<double> createBiasVector(int &numOfNeurons);
};