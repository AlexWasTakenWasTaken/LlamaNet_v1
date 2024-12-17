#pragma once

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <cmath>
#include <random>
#include <ctime>

class Neuron {

private: 
	std::vector<double> weights;
	double bias;

	double fRand(double min, double max); //generates a random number between -1 and 1
	double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

public:
    Neuron(int numInputs);
	double feedForward(const std::vector<double>& inputs);

	double activate(double x);
	double derivative(double x);

	std::vector<double> getWeights() const;
	double getWeightAtIndex(int index) const;
	double getBias() const;
};

#endif // !NEURON_H