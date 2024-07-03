#include "NNInput.h"
#include <cmath>
#include <iostream>
#include "NNHelper.h"

std::vector<std::vector<double>> NN_Input::spiral_data(int points, int classes, std::vector<int> &y)
{
    pointsNum = points;
    classesNum = classes;

    std::vector<std::vector<double>> X(points * classes, std::vector<double>(2, 0.0));
    y.resize(points * classes);

    // Seed the random number generator
    srand(static_cast<unsigned>(time(0)));

    for (int class_number = 0; class_number < classes; ++class_number)
    {
        int start_ix = points * class_number;     // Starting index for this class
        int end_ix = points * (class_number + 1); // Ending index for this class

        // Generate radius starting from a small positive value to avoid zero
        std::vector<double> r(points);
        for (int i = 0; i < points; ++i)
            r[i] = 0.01 + (static_cast<double>(i) / (points - 1)) * 0.99; // Avoid 0, starts from 0.01

        // Generate angle
        std::vector<double> t(points);
        double start_angle = class_number * 4.0;
        double end_angle = (class_number + 1) * 4.0;
        for (int i = 0; i < points; ++i)
            t[i] = start_angle + (static_cast<double>(i) / (points - 1)) * (end_angle - start_angle) + randomGenerator(-1.0, 1.0);

        // Populate X
        for (int i = start_ix; i < end_ix; ++i)
        {
            X[i][0] = r[i - start_ix] * sin(t[i - start_ix] * 2.5);
            X[i][1] = r[i - start_ix] * cos(t[i - start_ix] * 2.5);
        }

        // Populate y
        for (int i = start_ix; i < end_ix; ++i)
        {
            y[i] = class_number;
        }
    }

    input = X;
    trueInput = y;
    return X;
}
