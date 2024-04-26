#include <random>
#include <iomanip>



double randomGenerator(const double &minValue, const double &maxValue);

void printMatrix(const std::vector<std::vector<double>> matrix_);

void printVector(const std::vector<double> vector_);

void printVector(const std::vector<int> vector_);

std::vector<double> operator*(std::vector<std::vector<double>> matrix_, std::vector<double> vector_);

std::vector<std::vector<double>> operator*(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2);

std::vector<double> operator+(std::vector<double> vec, std::vector<double> values);

std::vector<std::vector<double>> operator+(std::vector<std::vector<double>> matrix, std::vector<double> vector_);

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix_);

int getNumOfRows(const std::vector<std::vector<double>> &matrix);

int getNumOfColumns(const std::vector<std::vector<double>> &matrix);

std::vector<std::vector<double>> clipValues(std::vector<std::vector<double>> &matrix);