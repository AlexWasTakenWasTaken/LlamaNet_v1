#include "Network.h"

Network::Network(std::vector<int> topology) {
	for (int i = 0; i < topology.size(); i++) {
		if (i == 0) {
			continue;
		}
		else {
			layers.push_back(new Layer(topology[i], topology[i - 1]));
		}
	}
}

std::vector<double> Network::predict(const std::vector<double>& inputData) {
	std::vector<double> outputs = inputData;
	for (Layer* layer : layers) {
		outputs = layer->feedForward(outputs);
	}
	return outputs;
}