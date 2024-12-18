#include <iostream>
#include <chrono>

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

int main() {
	std::srand(static_cast<unsigned>(std::time(0)));

	const auto startTime = std::chrono::steady_clock::now();

	std::vector<int> topology = { 3, 5, 8 };

	ActivationFunction* activationFunction = new LeakyReLU(0.01);
	Network* network = new Network(topology, activationFunction); 
	
	Network::NetworkParameters networkParameter;
	networkParameter.epochs = 10000;

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
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
		{0, 0, 0, 1, 0, 0, 0, 0},
		{0, 0, 0, 0, 1, 0, 0, 0},
		{0, 0, 0, 0, 0, 1, 0, 0},
		{0, 0, 0, 0, 0, 0, 1, 0},
		{0, 0, 0, 0, 0, 0, 0, 1} };

    network->train(inputs, targets, networkParameter);

    std::vector<double> output = network->frontpropogate(inputs[5]);

    for (double value : output) {
        std::cout << value << std::endl;
    }

    std::cout << std::endl;

    const auto endTime = std::chrono::steady_clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;


    return 0;
}