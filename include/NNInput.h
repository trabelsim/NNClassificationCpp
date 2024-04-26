#include <vector>

class NN_Input
{
    public:
        NN_Input(){};
        std::vector<std::vector<double>> getInput() { return input; };

        std::vector<std::vector<double>> spiral_data(int points, int classes);

        std::vector<int> getTrueInput() { return trueInput; };

    private:
        /*
        Size of the matrix is num_of_Samples x num_of_features
    */
        std::vector<std::vector<double>> defaultInputs = {
            // all samples have 4 features.
            {1, 2, 3, 2.5},        // 1st sample
            {2, 5, -1, 2},         // 2nd sample
            {-1.5, 2.7, 3.3, -0.8} // 3rd sample

        };

        std::vector<std::vector<double>> input;
        std::vector<int> trueInput;
};

