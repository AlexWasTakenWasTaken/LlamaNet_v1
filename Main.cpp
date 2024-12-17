#include <iostream>

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

int main() {
	std::srand(static_cast<unsigned>(std::time(0)));


	std::vector<int> topology = {3, 15, 15, 4};

	Network network(topology);


	std::vector<double> input = { 1.5, -1.2, 1.1 };

	std::vector<double> output = network.predict(input);

	for (double value : output) {
		std::cout << value << std::endl;
	}

	return 0;
}