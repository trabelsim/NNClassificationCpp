#include <vector>

class NN_Input
{
    public:
        NN_Input(){};
        std::vector<std::vector<float>> getInput() { return defaultInputs; }

        typedef struct
        {
            double *x; /* Holds the x y axis data. Data is formated x y x y x y*/
            double *y; /* Holds the group the data belongs too. Two steps of x is a single step of y*/
        } spiral_data_t;

        std::vector<std::vector<float>> spiral_data(int points, int classes);

    private:
        /*
        Size of the matrix is num_of_Samples x num_of_features
    */
        std::vector<std::vector<float>> defaultInputs = {
            // all samples have 4 features.
            {1, 2, 3, 2.5},        // 1st sample
            {2, 5, -1, 2},         // 2nd sample
            {-1.5, 2.7, 3.3, -0.8} // 3rd sample

        };
};

