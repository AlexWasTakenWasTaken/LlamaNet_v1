#pragma once

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <random>
#include <ctime>
#include "ActivationFunction.h"

class Neuron {

private: 
	std::vector<double> weights;
	double bias;
	ActivationFunction* activationFunction;

	double xavRand(int inputs, int outputs); //uses Xavier initialization method to determine the range of random weights
public:
    Neuron(int numInputs, int numOutputs, ActivationFunction* activationFunction);

	double dotProduct(const std::vector<double>& a, const std::vector<double>& b); //computes the dot product of two vectors

	double feedForward(const std::vector<double>& inputs);

	double activate(double x);

	std::vector<double> getWeights() const;
	double getWeightAtIndex(int index) const;
	double getBias() const;

	void setWeights(const std::vector<double>& newWeights);
	void setWeightAtIndex(int index, double newWeight);
	void setBias(double newBias);
};

#endif // !NEURON_H