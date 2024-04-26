#include "NNHelper.h"
#include <iostream>
#include <cmath>

using namespace std;

//Const values
const double eulerConst = std::exp(1.0);
const double minValueClipCatCrossEntropy = 1e-7; // almost 0
const double maxValueClipCatCrossEntropy = 1 - 1e-7; // almost 1

double randomGenerator(const double &minValue, const double &maxValue)
{
    std::mt19937_64 rng{};
    rng.seed(std::random_device{}());
    return std::uniform_real_distribution<>{minValue, maxValue}(rng);
}

std::vector<double> operator *(std::vector<std::vector<double>> matrix_, std::vector<double> vector_)
{
    std::vector<double> output{};
    for (auto &&matrixVec : matrix_)
    {
        if (matrixVec.size() != vector_.size())
        {
            cout << "E: DOT PRODUCT: Matrix vector size (" << matrixVec.size() << ") does not match vector size (" << vector_.size() << ")." << endl;

            return output;
        }

        double matrixVecOut{0};
        for (int i = 0; i < matrixVec.size(); i++)
        {
            matrixVecOut += matrixVec[i] * vector_[i];
        }

        output.push_back(matrixVecOut);
    }

    return output;
}

std::vector<std::vector<double>> operator*(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2)
{
    auto n1 = matrix1.size();
    auto m1 = matrix1[0].size();
    auto n2 = matrix2.size();
    auto m2 = matrix2[0].size();

    std::vector<std::vector<double>> output(n1, std::vector<double>(m2));

    if (m1 != n2)
    {
        cout << "E: DOT PRODUCT: Matrix vector size (" << n1 << " x " << m1 << ")"
             << ") does not match second matrix size (" << n2 << " x " << m2 << ")" << endl;

        return output;
    }

    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < m2; j++)
        {
            output[i][j] = 0;

            for (int k = 0; k < n2; k++)
            {
                output[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return output;
}

/*
    Method for adding values of two vectors together.
*/
std::vector<double> operator+(std::vector<double> vec, std::vector<double> values)
{
    std::vector<double> output{};

    if (vec.size() != values.size())
    {
        cout << "E: ADD: Vector size (" << vec.size() << ") does not match second vector size (" << values.size() << ")." << endl;

        return output;
    }

    for (int i = 0; i < vec.size(); i++)
    {
        output.push_back(vec[i] + values[i]);
    }

    return output;
}

/*
    Method for adding values of two vectors together.
*/
std::vector<std::vector<double>> operator+(std::vector<std::vector<double>> matrix, std::vector<double> vector_)
{
    std::vector<std::vector<double>> output{};

    if (matrix[0].size() != vector_.size())
    {
        cout << "E: ADD: Matrix size (" << matrix.size() << " x " << matrix[0].size()
             << ") does not match second vector size (" << vector_.size() << ")." << endl;

        return output;
    }

    for (auto &&row : matrix)
    {
        output.push_back(row + vector_);
    }

    return output;
}

void printMatrix(const std::vector<std::vector<double>> matrix_)
{
    auto n = matrix_.size();
    auto m = matrix_[0].size();
    cout << "Matrix size: " << n << " x " << m << endl;

    for (auto &&row : matrix_)
    {
        for (auto &&el : row)
        {
            cout << el << " ";
        }

        cout << endl;
    }
    cout << endl;
}

void printVector(const std::vector<double> vector_)
{
    auto n = vector_.size();
    cout << "Vector size: " << 1 << " x " << n << endl;

    for (auto &&el : vector_)
    {
        cout << el << " ";
    }
    cout << endl
         << endl;
}

void printVector(const std::vector<int> vector_)
{
    auto n = vector_.size();
    cout << "Vector size: " << 1 << " x " << n << endl;

    for (auto &&el : vector_)
    {
        cout << el << " ";
    }
    cout << endl
         << endl;
}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix_)
{
    auto n = matrix_.size();
    auto m = matrix_[0].size();

    std::vector<std::vector<double>> output(m, std::vector<double>(n));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            output[j][i] = matrix_[i][j];
        }
    }

    return output;
}


int getNumOfRows(const std::vector<std::vector<double>> &matrix)
{
    return matrix.size();
}

int getNumOfColumns(const std::vector<std::vector<double>> &matrix)
{
    return matrix[0].size();
}

std::vector<std::vector<double>> clipValues(std::vector<std::vector<double>> &matrix)
{

    for (auto &&row : matrix)
    {
        for (auto &&el : row)
        {
            if (el <= minValueClipCatCrossEntropy)
            {
                el = minValueClipCatCrossEntropy;
            }
            else if(el >= maxValueClipCatCrossEntropy)
            {
                el = maxValueClipCatCrossEntropy;
            }
        }
    }

    return matrix;
}
