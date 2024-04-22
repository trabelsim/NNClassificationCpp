#include <random>



float randomGenerator(const float &minValue, const float &maxValue);

void printMatrix(const std::vector<std::vector<float>> matrix_);

void printVector(const std::vector<float> vector_);

std::vector<float> operator*(std::vector<std::vector<float>> matrix_, std::vector<float> vector_);

std::vector<std::vector<float>> operator*(std::vector<std::vector<float>> matrix1, std::vector<std::vector<float>> matrix2);

std::vector<float> operator+(std::vector<float> vec, std::vector<float> values);

std::vector<std::vector<float>> operator+(std::vector<std::vector<float>> matrix, std::vector<float> vector_);

std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> matrix_);


/*
    Activation functions

*/
std::vector<std::vector<float>> activationReLU_forward(std::vector<std::vector<float>> &matrix);

std::vector<std::vector<float>> activationSoftMax_forward(std::vector<std::vector<float>> &matrix);
