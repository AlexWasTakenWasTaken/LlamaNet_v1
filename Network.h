#pragma once

#ifndef Network_H
#define Network_H

#include <vector>
#include "Layer.h"

class Network {
	struct NetworkParameters;
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

	void updateWeightsAndBiases(const Gradients& gradients, const NetworkParameters& param);

	void trainBatch(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<double>>& batchTargets, const NetworkParameters& param);

public:
	struct NetworkParameters {
		int epochs = 1;
		int batchSize = 1;

		double learningRate = 0.1;
		double beta1 = 0.9;
		double beta2 = 0.999;
		double epsilon = 1e-8;
	};

	Network(std::vector<int> topology, ActivationFunction* activationFunction);
	std::vector<double> frontpropogate(const std::vector<double>& inputData);


	void train(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& targets, const NetworkParameters& param);
};
#endif // !Network_H