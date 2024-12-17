#include "Layer.h"

Layer::Layer(int numNeurons, int numInputs) {
	for (int i = 0; i < numNeurons; i++) {
		neurons.push_back(Neuron(numInputs, numNeurons));
	}
}

std::vector<double> Layer::feedForward(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    for (Neuron neuron : neurons) {
        outputs.push_back(neuron.feedForward(inputs));
    }
    return outputs;
}

int Layer::getNumNeurons() const {
	return neurons.size();
}

Neuron Layer::getNeuronAtIndex(int index) const {
	return neurons[index];
}