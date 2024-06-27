#include <random>
#include <iomanip>



double randomGenerator(const double &minValue, const double &maxValue);

void printMatrix(const std::vector<std::vector<double>> matrix_);

void printVector(const std::vector<double> vector_);

void printVector(const std::vector<int> vector_);

std::vector<double> operator*(std::vector<std::vector<double>> matrix_, std::vector<double> vector_);

std::vector<std::vector<double>> operator*(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2);

std::vector<std::vector<double>> operator*(double scalar, std::vector<std::vector<double>> matrix);

std::vector<double> operator*(double scalar, std::vector<double> vector);

// std::vector<std::vector<double>> operator/(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2);

std::vector<std::vector<double>> operator/(std::vector<std::vector<double>> matrix1, int scalar);

std::vector<double> operator+(std::vector<double> vec, std::vector<double> values);

std::vector<double> operator-(std::vector<double> vec1, std::vector<double> vec2);

std::vector<std::vector<double>> operator+(std::vector<std::vector<double>> matrix, std::vector<double> vector_);

std::vector<std::vector<double>> operator+(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2);

std::vector<std::vector<double>> operator-(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2);

std::vector<std::vector<double>> operator-(std::vector<std::vector<double>> matrix1, double scalar);

std::vector<double> operator-(std::vector<double> vector1, double scalar);

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix_);

/* Sum the columns of a provided matrix*/
std::vector<double> sumElementsOnAxisZero(std::vector<std::vector<double>>& matrix);

int getNumOfRows(const std::vector<std::vector<double>> &matrix);

int getNumOfColumns(const std::vector<std::vector<double>> &matrix);

std::vector<std::vector<double>> clipValues(std::vector<std::vector<double>> &matrix);

std::vector<std::vector<double>> createDiagonalMatrix(const std::vector<double>& vector);

std::vector<std::vector<double>> createIdentityMatrix(int size);

std::vector<std::vector<double>> sLToOneHotEncodedL(const std::vector<int> &yTrue, int labels);

std::vector<int> oneHotEncodedToDiscrete(const std::vector<std::vector<int>> &yTrue);
