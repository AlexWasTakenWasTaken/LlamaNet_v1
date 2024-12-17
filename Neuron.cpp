#include "Neuron.h"

double Neuron::xavRand(int input, int output)
{
	double max = sqrt(6.0 / (input + output));

    double f = (double)rand() / RAND_MAX;
    return f * (2 * max) - max;
}

double Neuron::dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
	double sum = 0;
	for (int i = 0; i < a.size(); i++) {
		sum += a[i] * b[i];
	}
	return sum;
}

Neuron::Neuron(int numInputs, int numOutputs){
    std::srand(static_cast<unsigned>(std::time(0)));

    for (int i = 0; i < numInputs; ++i) {
        weights.push_back(xavRand(numInputs, numOutputs));
    }
    bias = xavRand(numInputs, numOutputs);
}

double Neuron::feedForward(const std::vector<double>& inputs) {
	return activate(dotProduct(inputs, weights));
}

double Neuron::activate(double x) {
	return (x > 0 ? x : 0);
}

double Neuron::derivative(double x) {
	return (x > 0 ? 1 : 0);
}

std::vector<double> Neuron::getWeights() const {
	return weights;
}

double Neuron::getWeightAtIndex(int index) const {
	return weights[index];
}

double Neuron::getBias() const {
	return bias;
}

void Neuron::setWeights(const std::vector<double>& newWeights) {
	weights = newWeights;
}

void Neuron::setWeightAtIndex(int index, double newWeight) {
	weights[index] = newWeight;
}

void Neuron::setBias(double newBias) {
	bias = newBias;
}