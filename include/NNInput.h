#include <vector>

class NN_Input
{
    public:
        NN_Input(){};
        std::vector<std::vector<double>> getInput() { return input; };

        std::vector<std::vector<double>> spiral_data(int points, int classes, std::vector<int> &y);

        std::vector<int> getTrueInput() { return trueInput; };

        int getNumPoints() { return pointsNum; }
        int getNumClasses() { return classesNum; }

    private:
        std::vector<std::vector<double>> input;
        std::vector<int> trueInput;
        int pointsNum;
        int classesNum;
};

