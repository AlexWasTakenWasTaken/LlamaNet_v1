#include "Neuron.h"

Neuron::Neuron(int numInputs){
    std::srand(static_cast<unsigned>(std::time(0)));

    for (int i = 0; i < numInputs; ++i) {
        weights.push_back(fRand(-1, 1));
    }
    bias = fRand(-0.1 , 0.1);
}

double Neuron::fRand(double min, double max)
{
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

double Neuron::feedForward(const std::vector<double>& inputs) {
	double sum = 0;
	for (int i = 0; i < weights.size(); i++) {
		sum += weights[i] * inputs[i];
	}
	sum += bias;
	return activate(sum);
}