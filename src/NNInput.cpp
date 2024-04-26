#include "NNInput.h"
#include <cmath>
#include <iostream>
#include "NNHelper.h"

std::vector<std::vector<double>> NN_Input::spiral_data(int points, int classes)
{
    this->points = points;
    this->classes = classes;

    std::vector<std::vector<double>> X(points * classes, std::vector<double>(2, 0.0f));
    std::vector<int> y(points * classes, 0);

    for (int class_number = 0; class_number < classes; ++class_number)
    {
        int start_ix = points * class_number;
        int end_ix = points * (class_number + 1);

        // Generate radius
        std::vector<double> r;
        for (int i = 0; i < points; ++i)
            r.push_back(static_cast<double>(i) / (points - 1));

        // Generate angle
        std::vector<double> t;
        double start_angle = class_number * 4.0f;
        double end_angle = (class_number + 1) * 4.0f;
        for (int i = 0; i < points; ++i)
            t.push_back(start_angle + (static_cast<double>(i) / (points - 1)) * (end_angle - start_angle) + randomGenerator(-1.0,1.0));

        // Populate X
        for (int i = start_ix; i < end_ix; ++i)
        {
            X[i][0] = r[i - start_ix] * sin(t[i - start_ix] * 2.5f);
            X[i][1] = r[i - start_ix] * cos(t[i - start_ix] * 2.5f);
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
