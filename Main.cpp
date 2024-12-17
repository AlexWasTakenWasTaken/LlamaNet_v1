#include <iostream>
#include <chrono>

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

int main() {
	std::srand(static_cast<unsigned>(std::time(0)));

    const auto startTime = std::chrono::steady_clock::now();

    std::vector<int> topology = { 3, 15, 15, 15, 8 };
    Network network(topology);

    std::vector<double> input = { 1, 0, 0 };
    std::vector<double> target = { 0, 0, 0, 1, 0, 0, 0, 0 };


    for (int i = 0; i < 1000000; i++) {
        network.backpropogate(input, target);
    }

    std::vector<double> output = network.frontpropogate(input);

    for (double value : output) {
        std::cout << value << std::endl;
    }

    std::cout << std::endl;

    const auto endTime = std::chrono::steady_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;

    return 0;
}