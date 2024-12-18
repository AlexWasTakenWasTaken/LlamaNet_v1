#include <iostream>
#include <chrono>

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

int main() {
	std::srand(static_cast<unsigned>(std::time(0)));

    const auto startTime = std::chrono::steady_clock::now();

    std::vector<int> topology = { 3, 15, 13, 1 };
    Network network(topology);

    std::vector<std::vector<double>> inputs = { 
        {1, 1, 1}, 
        {1, 1, 0}, 
        {1, 0, 1},
		{1, 0, 0},
		{0, 1, 1},
		{0, 1, 0},
		{0, 0, 1},
		{0, 0, 0} };

    std::vector<std::vector<double>> targets = { 
        {7}, 
        {6}, 
        {5}, 
        {4}, 
        {3}, 
        {2}, 
        {1}, 
        {0} };

    network.trainBatch(inputs, targets, 1000, 0.1);

    std::vector<double> output = network.frontpropogate(inputs[0]);

    for (double value : output) {
        std::cout << value << std::endl;
    }

    std::cout << std::endl;

    const auto endTime = std::chrono::steady_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;


    return 0;
}