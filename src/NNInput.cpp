#include "NNInput.h"
#include <cmath>
#include <iostream>
#include "NNHelper.h"

std::vector<std::vector<float>> NN_Input::spiral_data(int points, int classes)
{
    std::vector<std::vector<float>> X(points * classes, std::vector<float>(2, 0.0f));
    std::vector<int> y(points * classes, 0);

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
            t.push_back(start_angle + (static_cast<float>(i) / (points - 1)) * (end_angle - start_angle) + randomGenerator(-1.0,1.0));

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

// std::vector<std::vector<float>> NN_Input::spiral_data(int points, int classes)
// {
//     std::vector<std::vector<float>> X(points, std::vector<float>(2));
//     float angle_step = 2.0 * 3.14159265358979323846 / (float)classes;

//     for (int i = 0; i < points; ++i)
//     {
//         float r = (float)i / (float)points * 5.0f;
//         float angle = i * angle_step;
//         float x1 = r * sin(angle);
//         float x2 = r * cos(angle);
//         X[i][0] = x1;
//         X[i][1] = x2;
//     }

//     return X;
// }