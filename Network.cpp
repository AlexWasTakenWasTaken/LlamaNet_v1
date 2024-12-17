#include "Network.h"

// Huber Loss Function for cost calculation
double Network::costFunction(const std::vector<double>& target, const std::vector<double>& output) {
	double cost = 0;

	// default delta: 1
	const double DELTA = 1;

	for (int i = 0; i < target.size(); i++) {
		double error = target[i] - output[i];

		if (abs(error) < DELTA) {
			cost += 0.5 * error * error;
		}
		else {
			cost += DELTA * (abs(error) - 0.5 * DELTA);
		}
	}
	return cost / target.size();
}

double Network::derivativeCost_Output(double target, double output) {
	double gradient = 0;

	// default delta: 1
	const double DELTA = 1;

	double error = target - output;

	// derivative of the respective Huber loss functions
	if (abs(error) < DELTA) {
			gradient = error;
	}
	else {
			
		gradient = DELTA * (error > 0 ? 1 : -1);
		
	}
	return gradient;
}

std::vector<double> Network::gradientCalculation(const std::vector<double>& target, const std::vector<double>& output) {
	std::vector<double> gradients(output.size());

	for (int i = 0; i < target.size(); i++) {
		gradients[i] = derivativeCost_Output(target[i], output[i]);
	}
	return gradients;
}

Network::Network(std::vector<int> topology) {
	for (int i = 1; i < topology.size(); i++) {
		layers.push_back(new Layer(topology[i], topology[i - 1]));
	}
}

std::vector<double> Network::frontpropogate(const std::vector<double>& inputData) {
	std::vector<double> outputs = inputData;
	for (Layer* layer : layers) {
		outputs = layer->feedForward(outputs);
	}
	return outputs;
}