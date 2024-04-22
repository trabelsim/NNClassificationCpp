#include "NNHelper.h"
#include <iostream>
#include <cmath>

using namespace std;

//Const values
const double eulerConst = std::exp(1.0);

float randomGenerator(const float &minValue, const float &maxValue)
{
    std::mt19937_64 rng{};
    rng.seed(std::random_device{}());
    return std::uniform_real_distribution<>{minValue, maxValue}(rng);
}

std::vector<float> operator *(std::vector<std::vector<float>> matrix_, std::vector<float> vector_)
{
    std::vector<float> output{};
    for (auto &&matrixVec : matrix_)
    {
        if (matrixVec.size() != vector_.size())
        {
            cout << "E: DOT PRODUCT: Matrix vector size (" << matrixVec.size() << ") does not match vector size (" << vector_.size() << ")." << endl;

            return output;
        }

        float matrixVecOut{0};
        for (int i = 0; i < matrixVec.size(); i++)
        {
            matrixVecOut += matrixVec[i] * vector_[i];
        }

        output.push_back(matrixVecOut);
    }

    return output;
}

std::vector<std::vector<float>> operator*(std::vector<std::vector<float>> matrix1, std::vector<std::vector<float>> matrix2)
{
    auto n1 = matrix1.size();
    auto m1 = matrix1[0].size();
    auto n2 = matrix2.size();
    auto m2 = matrix2[0].size();

    std::vector<std::vector<float>> output(n1, std::vector<float>(m2));

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
std::vector<float> operator+(std::vector<float> vec, std::vector<float> values)
{
    std::vector<float> output{};

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
std::vector<std::vector<float>> operator+(std::vector<std::vector<float>> matrix, std::vector<float> vector_)
{
    std::vector<std::vector<float>> output{};

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

void printMatrix(const std::vector<std::vector<float>> matrix_)
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

void printVector(const std::vector<float> vector_)
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

std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> matrix_)
{
    auto n = matrix_.size();
    auto m = matrix_[0].size();

    std::vector<std::vector<float>> output(m, std::vector<float>(n));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            output[j][i] = matrix_[i][j];
        }
    }

    return output;
}

std::vector<std::vector<float>> activationReLU_forward(std::vector<std::vector<float>> &matrix)
{
    for(auto &&row : matrix)
    {
        for(auto &&el : row)
        {
            if(el < 0)
            {
                el = 0;
            }
        }
    }

    return matrix;
}

int getNumOfRows(const std::vector<std::vector<float>> &matrix)
{
    return matrix.size();
}

int getNumOfColumns(const std::vector<std::vector<float>> &matrix)
{
    return matrix[0].size();
}

std::vector<std::vector<float>> activationSoftMax_forward(std::vector<std::vector<float>> &matrix)
{
    int n = getNumOfRows(matrix);
    int m = getNumOfColumns(matrix);

    std::vector<std::vector<float>> outputMatrix(n, std::vector<float>(m));


    //First calculate the exponential value of each matrix-element. => e^(value in place i,j)
    for(int i=0; i< n; i++)
    {
        for(int j=0; j<m; j++)
        {
            outputMatrix[i][j] = pow(eulerConst,matrix[i][j]);
        }
    }

    // Next calculate the overall sum of all the element (which values have been exponentiated)
    double expSum = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            expSum += outputMatrix[i][j];
        }
    }

    // Finally we divide each element of the matrix by the calculated expSum.
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            outputMatrix[i][j] = outputMatrix[i][j] / expSum;
        }
    }

    return outputMatrix;

}