#pragma once

#ifndef Network_H
#define Network_H

#include <vector>
#include "Layer.h"

class Network {
private:

	struct Gradients {
		std::vector<std::vector<std::vector<double>>> weightGradients; //layer, neuron, weight
		std::vector<std::vector<double>> biasGradients; //layer, neuron
	};

	std::vector<Layer*> layers;
	ActivationFunction* activationFunction;

	std::vector<std::vector<std::vector<double>>> mWeights;
	std::vector<std::vector<std::vector<double>>> vWeights;
	std::vector<std::vector<double>> mBiases;
	std::vector<std::vector<double>> vBiases;
	int t;

	double costFunction(const std::vector<double>& target, const std::vector<double>& output);
	double costFunctionDerivative(double target, double output);

	// deriivative helper functions
	double derivativeActivation_Z(double z);
	std::vector<double> derivativeCost_Output(const std::vector<double>& target, const std::vector<double>& output);

	Gradients computeGradients(const std::vector<double>& inputData, const std::vector<double>& target);
	void updateWeightsAndBiases(const Gradients& gradients, double learningRate, int batchSize);

	void updateWeightsAndBiasesAdam(const Gradients& gradients, double learningRate, int batchSize, double beta1, double beta2, double epsilon);

public:
	Network(std::vector<int> topology, ActivationFunction* activationFunction);
	std::vector<double> frontpropogate(const std::vector<double>& inputData);

	void trainBatch(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<double>>& batchTargets, double learningRate, double beta1, double beta2, double epsilon);
	void train(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& targets, int epochs, double learningRate, int batchSize, double beta1, double beta2, double epsilon);
};
#endif // !Network_H