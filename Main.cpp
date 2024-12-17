#include <iostream>

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

int main() {

	std::vector<int> topology = { 3, 4, 2 };

	Network network(topology);


	std::vector<double> input = { 1.5, -0.2, 0.1 };

	std::vector<double> output = network.predict(input);

	for (double value : output) {
		std::cout << value << std::endl;
	}

	return 0;
}