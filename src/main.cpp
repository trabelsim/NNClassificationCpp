#include <iostream>
#include "NNLayer.h"
#include "NNInput.h"
#include "NNHelper.h"

using namespace std;

int main()
{
    cout << "NNGen" << endl;

    // Prepare data

    auto nnInput = NN_Input();
    auto spiralInput = nnInput.spiral_data(100,3);

    auto layer1 = NN_Layer_Dense(2, 3);
    auto layer2 = NN_Layer_Dense(3,3);


    //Going forward
    auto l1Output = layer1.forward(spiralInput);

    auto reLUOutput = activationReLU_forward(l1Output); // max (0, value)
    
    auto l2Output = layer2.forward(reLUOutput);

    auto softMaxOut = activationSoftMax_forward(l2Output); // probabilities

    printMatrix(softMaxOut);


    return 0;
}