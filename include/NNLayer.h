#include <vector>
#include "NNHelper.h"

class NN_Layer_Dense
{
    public:
        NN_Layer_Dense(int numOfInputFeatures, int numOfNeurons); // Steps to perform: 1)know the size of the input and the num of neurons we want to create, 2) normalize the data,

        std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &input);

        std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &dValues);
        /*
            Getters
        */
        std::vector<std::vector<double>> getWeights();
        std::vector<double> getBias();
        std::vector<std::vector<double>> getOutput();

    private:
        // Forward vars.
        std::vector<std::vector<double>> weights_;
        std::vector<double> bias_;
        std::vector<std::vector<double>> output_;

        //Backward vars.
        std::vector<std::vector<double>> input_; // stored to calculate the derivative during backpropagation
        std::vector<std::vector<double>> dWeights_;
        std::vector<std::vector<double>> dBiases_;
        std::vector<std::vector<double>> dInputs_;

        std::vector<std::vector<double>> createWeightsMatrix(int &numOfInputFeatures, int &numOfNeurons);
        std::vector<double> createBiasVector(int &numOfNeurons);
};