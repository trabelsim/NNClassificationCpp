#include "NNInput.h"
#include <cmath>
#include <iostream>
#include "NNHelper.h"

std::vector<std::vector<float>> NN_Input::spiral_data(int points, int classes)
{
    std::vector<std::vector<float>> X(points * classes, std::vector<float>(2, 0.0f));
    std::vector<int> y(points * classes, 0);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 0.2);

    for (int class_number = 0; class_number < classes; ++class_number)
    {
        int start_ix = points * class_number;
        int end_ix = points * (class_number + 1);

        // Generate radius
        std::vector<float> r;
        for (int i = 0; i < points; ++i)
            r.push_back(static_cast<float>(i) / (points - 1));

        // Generate angle
        std::vector<float> t;
        float start_angle = class_number * 4.0f;
        float end_angle = (class_number + 1) * 4.0f;
        for (int i = 0; i < points; ++i)
            t.push_back(start_angle + (static_cast<float>(i) / (points - 1)) * (end_angle - start_angle) + dis(gen));

        // Populate X
        for (int i = start_ix; i < end_ix; ++i)
        {
            X[i][0] = r[i - start_ix] * sin(t[i - start_ix] * 2.5f);
            X[i][1] = r[i - start_ix] * cos(t[i - start_ix] * 2.5f);
        }

        // Populate y
        for (int i = start_ix; i < end_ix; ++i)
            y[i] = class_number;
    }

    return X;
}
