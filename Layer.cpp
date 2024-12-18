#include "Layer.h"

Layer::Layer(int numNeurons, int numInputs) {
	for (int i = 0; i < numNeurons; i++) {
		neurons.push_back(new Neuron(numInputs, numNeurons));
	}
}

std::vector<double> Layer::feedForward(const std::vector<double>& inputs) {
    std::vector<double> outputs;
    this->zValues.clear();

    for (Neuron* neuron : neurons) {
        // z = dotProduct(inputs, weights) + bias
        double z = neuron->dotProduct(inputs, neuron->getWeights()) + neuron->getBias();
        zValues.push_back(z);

        outputs.push_back(neuron->activate(z));
    }

	this->aValues = outputs;
    return outputs;
}

std::vector<Neuron*> Layer::getNeurons() const {
	return neurons;
}

std::vector<double> Layer::getZValues() const {
	return aValues;
}

std::vector<double> Layer::getAValues() const {
	return aValues;
}