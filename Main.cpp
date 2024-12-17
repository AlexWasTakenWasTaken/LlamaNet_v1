#include <iostream>

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

int main() {
	std::srand(static_cast<unsigned>(std::time(0)));


	std::vector<int> topology = {3, 15, 15, 8};

	Network network(topology);


	std::vector<double> input = { 1.0, 0, 0 };

	std::vector<double> output = network.predict(input);

	for (double value : output) {
		std::cout << value << std::endl;
	}

	std::cout << std::endl;

	return 0;
}