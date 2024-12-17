#pragma once

#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include "Neuron.h"

class Layer {

private:
	std::vector<Neuron*> neurons;

	std::vector<double> zValues;
	std::vector<double> aValues;
	
public:
	Layer(int numNeurons, int numInputs);
	std::vector<double> feedForward(const std::vector<double>& inputs);

	int getNumNeurons() const;

	Neuron* getNeuronAtIndex(int index) const;

	std::vector<Neuron*> getNeurons() const;
	std::vector<double> getZValues() const;
	std::vector<double> getAValues() const;
};

#endif // !LAYER_H