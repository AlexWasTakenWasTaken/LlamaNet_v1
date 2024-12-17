#pragma once

#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include "Neuron.h"

class Layer {

private:
	std::vector<Neuron*> neurons;
public:
	Layer(int numNeurons, int numInputs);
	std::vector<double> feedForward(const std::vector<double>& inputs);

	int getNumNeurons() const;
	Neuron* getNeuronAtIndex(int index) const;
};

#endif // !LAYER_H