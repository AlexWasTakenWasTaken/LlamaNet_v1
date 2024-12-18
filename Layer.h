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
	Layer(int numNeurons, int numInputs, ActivationFunction* activationFunction);
	std::vector<double> feedForward(const std::vector<double>& inputs);

	std::vector<Neuron*> getNeurons() const;
	std::vector<double> getZValues() const;
	std::vector<double> getAValues() const;
};

#endif // !LAYER_H