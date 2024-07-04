#include "NNHelper.h"
#include <iostream>
#include <cmath>
#include <algorithm>

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

    
    if (m1 != n2)
    {
        cout << "E: DOT PRODUCT: Matrix vector size (" << n1 << " x " << m1 << ")"
             << ") does not match second matrix size (" << n2 << " x " << m2 << ")" << endl;

        return {};
    }

    std::vector<std::vector<double>> output(n1, std::vector<double>(m2, 0.0));

    for (int i = 0; i < n1; ++i)
    {
        for (int j = 0; j < m2; ++j)
        {
            for (int k = 0; k < m1; ++k)
            {
                output[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }

    return output;
}

std::vector<std::vector<double>> operator*(double scalar, std::vector<std::vector<double>> matrix)
{
    auto n1 = matrix.size();
    auto m1 = matrix[0].size();
    std::vector<std::vector<double>> output(n1, std::vector<double>(m1));

    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < m1; j++)
        {
            output[i][j] = matrix[i][j] * scalar;
        }
    }

    return output;
}

std::vector<double> operator*(double scalar, std::vector<double> vector)
{
    std::vector<double> output(vector.size(), 0.0);

    for (int i = 0; i < vector.size(); i++)
    {
        output[i] = vector[i] * scalar;
    }

    return output;
}

std::vector<std::vector<double>> operator/(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2)
{
    auto n1 = matrix1.size();
    auto m1 = matrix1[0].size();
    auto n2 = matrix2.size();
    auto m2 = matrix2[0].size();

    std::vector<std::vector<double>> output(n1, std::vector<double>(m1));

    if ((n1 != n2) || (m1 != m2))
    {
        cout << "E: DOT PRODUCT: Matrix vector size (" << n1 << " x " << m1 << ")"
             << ") does not match second matrix size (" << n2 << " x " << m2 << ")" << endl;

        return output;
    }

    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < m1; j++)
        {
            if (matrix2[i][j] == 0.0)
            {
                // Handle error: division by zero
                return {};
            }

            output[i][j] = matrix1[i][j] / matrix2[i][j];
        }
    }

    return output;
}

std::vector<std::vector<double>> operator/(std::vector<std::vector<double>> matrix1, int scalar)
{
    if(scalar != 0)
    {
        auto n1 = matrix1.size();
        auto m1 = matrix1[0].size();
        std::vector<std::vector<double>> output(n1, std::vector<double>(m1));

        for (int i = 0; i < n1; i++)
        {
            for (int j = 0; j < m1; j++)
            {
                output[i][j] = matrix1[i][j] / scalar;
            }
        }

        return output;
    }
    else
    {
        return{};
    }
    
}

std::vector<double> operator/(const std::vector<double> &vec, double scalar)
{
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
    {
        result[i] = vec[i] / scalar;
    }
    return result;
}

/*
    Method for adding values of two vectors together.
*/
std::vector<double> operator+(std::vector<double> vec, std::vector<double> values)
{
    std::vector<double> output(vec.size());

    if (vec.size() != values.size())
    {
        cout << "E: ADD: Vector size (" << vec.size() << ") does not match second vector size (" << values.size() << ")." << endl;

        return output;
    }

    for (int i = 0; i < vec.size(); i++)
    {
        output[i] = vec[i] + values[i];
    }

    return output;
}

std::vector<double> operator-(std::vector<double> vec1, std::vector<double> vec2)
{
    
    if (vec1.size() != vec2.size())
    {
        cout << "E: SUB: Vector size (" << vec1.size() << ") does not match second vector size (" << vec2.size() << ")." << endl;

        return {};
    }
    std::vector<double> output(vec2.size());

    for (int i = 0; i < vec1.size(); i++)
    {
        output[i] = vec1[i] - vec2[i];
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

std::vector<std::vector<double>> operator+(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2)
{
    auto n1 = matrix1.size();
    auto m1 = matrix1[0].size();
    auto n2 = matrix2.size();
    auto m2 = matrix2[0].size();

    std::vector<std::vector<double>> output(n1, std::vector<double>(m1));

    if ((n1 != n2) || (m1 != m2))
    {
        cout << "E: SUM: Matrix vector size (" << n1 << " x " << m1 << ")"
             << ") does not match second matrix size (" << n2 << " x " << m2 << ")" << endl;

        return output;
    }

    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < m1; j++)
        {
            output[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }

    return output;
}

std::vector<std::vector<double>> operator-(std::vector<std::vector<double>> matrix1, std::vector<std::vector<double>> matrix2)
{
    auto n1 = matrix1.size();
    auto m1 = matrix1[0].size();
    auto n2 = matrix2.size();
    auto m2 = matrix2[0].size();

    std::vector<std::vector<double>> output(n1, std::vector<double>(m1, 0.0));

    if ((n1 != n2) || (m1 != m2))
    {
        cout << "E: SUBTRACT: Matrix vector size (" << n1 << " x " << m1 << ")"
             << ") does not match second matrix size (" << n2 << " x " << m2 << ")" << endl;

        return output;
    }

    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < m1; j++)
        {
            output[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    return output;
}

std::vector<std::vector<double>> operator-(std::vector<std::vector<double>> matrix1, double scalar)
{
    auto n1 = matrix1.size();
    auto m1 = matrix1[0].size();
    std::vector<std::vector<double>> output(n1, std::vector<double>(m1));

    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < m1; j++)
        {
            output[i][j] = matrix1[i][j] - scalar;
        }
    }

    return output;
}

std::vector<double> operator-(std::vector<double> vector1, double scalar)
{
    auto n1 = vector1.size();
    std::vector<double> output(n1, 0.0);
    output = vector1;

    for (int i = 0; i < n1; i++)
    {
        output[i] = vector1[i] - scalar;
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

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> matrix)
{
    if(matrix.empty())
    {
        std::cout<< "ERROR TRANSPOSE" << std::endl;
        return {};
    }

    auto n = matrix.size();
    auto m = matrix[0].size();

    for (const auto &row : matrix)
    {
        if (row.size() != m)
        {
            std::cout << "ERROR: Matrix rows are not of equal length" << std::endl;
            return {};
        }
    }

    std::vector<std::vector<double>> output(m, std::vector<double>(n));

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            output[j][i] = matrix[i][j];
        }
    }

    return output;
}

std::vector<double> sumElementsOnAxisZero(std::vector<std::vector<double>>& matrix)
{
    if(matrix.empty())
    {
        std::cout << "ERROR sumElementsOnAxisZero" << std::endl;
        return {};
    }
    
    auto m = matrix[0].size();
    for (const auto &row : matrix)
    {
        if (row.size() != m)
        {
            std::cout << "ERROR: Matrix rows are not of equal length" << std::endl;
            return {};
        }
    }

    std::vector<double> sum(m, 0.0);
    for (const auto &row : matrix)
    {
        for(int j=0; j < m; ++j)
        {
            sum[j] += row[j];
        }
    }

    return sum;
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
    std::vector<std::vector<double>> output_ = matrix;

    for (auto &row : output_)
    {
        for (auto &el : row)
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

    return output_;
}

std::vector<std::vector<double>> createDiagonalMatrix(const std::vector<double> &vector)
{
    int vecSize = vector.size();
    std::vector<std::vector<double>> output(vecSize, std::vector<double>(vecSize));

    // Reset to 0 all values.
    for (auto &&row : output)
    {
        for (auto &&el : row)
        {
            el = 0;
        }
    }

    for(int i=0; i < vecSize; i++)
    {
        output[i][i] = vector[i];
    }

    return output;
}

std::vector<std::vector<double>> createIdentityMatrix(int size)
{
    std::vector<std::vector<double>> identityMatrix(size, std::vector<double>(size, 0.0));
    
    for (int i = 0; i < size; ++i)
    {
        identityMatrix[i][i] = 1.0;
    }
    return identityMatrix;
}

std::vector<std::vector<double>> sLToOneHotEncodedL(const std::vector<int> &yTrue, int labels)
{
    std::vector<std::vector<double>> oneHotEncodedLabels(yTrue.size(), std::vector<double>(labels, 0.0));

    for (size_t i = 0; i < yTrue.size(); ++i)
    {
        oneHotEncodedLabels[i][yTrue[i]] = 1.0;
    }
    return oneHotEncodedLabels;
}

std::vector<int> oneHotEncodedToDiscrete(const std::vector<std::vector<int>> &yTrue)
{
    std::vector<int> discreteValues;

    for(const auto& row : yTrue)
    {
        auto maxIterator = std::max_element(row.begin(), row.end());
        int index = std::distance(row.begin(), maxIterator);
        discreteValues.push_back(index);
    }

    return discreteValues;
}
