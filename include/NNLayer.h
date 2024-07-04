#ifndef NN_LAYER_H
#define NN_LAYER_H

#include <vector>
#include "NNHelper.h"

class NN_Layer_Dense
{
    public:
        // Steps to perform: 1)know the size of the input and the num of neurons we want to create, 2) normalize the data,
        NN_Layer_Dense(int numOfInputFeatures, int numOfNeurons); 

        std::vector<std::vector<double>> forward(std::vector<std::vector<double>> &input);

        std::vector<std::vector<double>> backward(std::vector<std::vector<double>> &dValues);
        /*
            Getters
        */
        std::vector<std::vector<double>>& getWeights();
        std::vector<double>& getBias();
        std::vector<std::vector<double>>& getOutput();

        // Backwards
        std::vector<std::vector<double>>& getdWeights();
        std::vector<double>& getdBias();
        std::vector<std::vector<double>>& getdInput();

        // Momentum
        std::vector<std::vector<double>>& getWeightsMomentum();
        std::vector<double>& getBiasMomentum();

        /*
            Setters
        */
        std::vector<std::vector<double>> setWeights(std::vector<std::vector<double>>& newWeights);
        std::vector<double> setBias(std::vector<double> &newBias);

        std::vector<std::vector<double>> setWeightsMomentum(std::vector<std::vector<double>> & newMomentum);
        std::vector<double> setBiasMomentum(std::vector<double>& newMomentum);

    private:
        // Forward vars.
        std::vector<std::vector<double>> weights_;
        std::vector<double> bias_;
        std::vector<std::vector<double>> output_;

        //Backward vars.
        std::vector<std::vector<double>> input_; // stored to calculate the derivative during backpropagation
        std::vector<std::vector<double>> dWeights_;
        std::vector<double> dBiases_;
        std::vector<std::vector<double>> dInputs_;

        std::vector<std::vector<double>> createWeightsMatrix(int &numOfInputFeatures, int &numOfNeurons);
        std::vector<double> createBiasVector(int &numOfNeurons);

        // Optimizer momentum
        void createWeightsMomentum();
        void createBiasMomentum();
        std::vector<std::vector<double>> weightsMomentum;
        std::vector<double> biasMomentum;
};
#endif //NN_LAYER_H