#include "Network.h"

// Huber Loss Function for cost calculation
double Network::costFunction(const std::vector<double>& target, const std::vector<double>& output) {
	double cost = 0;

	// default delta: 1
	const double DELTA = 1;

	for (int i = 0; i < target.size(); i++) {
		double error = target[i] - output[i];

		if (abs(error) < DELTA) {
			cost += 0.5 * error * error;
		}
		else {
			cost += DELTA * (abs(error) - 0.5 * DELTA);
		}
	}
	return cost / target.size();
}

double Network::costFunctionDerivative(double target, double output) {
	double gradient = 0;

	// default delta: 1
	const double DELTA = 1;

	double error = output - target;

	// derivative of the respective Huber loss functions
	if (abs(error) < DELTA) {
		gradient = error;
	}
	else {
		gradient = DELTA * (error > 0 ? 1 : -1);
	}
	return gradient;
}

double Network::derivativeActivation_Z(double z) {
	return activationFunction->derivative(z);
}

std::vector<double> Network::derivativeCost_Output(const std::vector<double>& target, const std::vector<double>& output) {
	std::vector<double> gradients(output.size());

	for (int i = 0; i < target.size(); i++) {
		gradients[i] = costFunctionDerivative(target[i], output[i]);
	}
	return gradients;
}

Network::Network(std::vector<int> topology, ActivationFunction* activationFunction) {
	for (int i = 1; i < topology.size(); i++) {
		layers.push_back(new Layer(topology[i], topology[i - 1], activationFunction));
	}
	this->activationFunction = activationFunction;

	mWeights.resize(layers.size());
	vWeights.resize(layers.size());
	mBiases.resize(layers.size());
	vBiases.resize(layers.size());

	for (int l = 0; l < layers.size(); l++) {
		Layer* layer = layers[l];
		int neuronCount = layer->getNeurons().size();
		mBiases[l].resize(neuronCount, 0.0);
		vBiases[l].resize(neuronCount, 0.0);

		mWeights[l].resize(neuronCount);
		vWeights[l].resize(neuronCount);

		for (int n = 0; n < neuronCount; n++) {
			int weightCount = layer->getNeurons()[n]->getWeights().size();
			mWeights[l][n].resize(weightCount, 0.0);
			vWeights[l][n].resize(weightCount, 0.0);
		}
	}

	t = 0;
}

std::vector<double> Network::frontpropogate(const std::vector<double>& inputData) {
	std::vector<double> outputs = inputData;
	for (Layer* layer : layers) {
		outputs = layer->feedForward(outputs);
	}
	return outputs;
}

Network::Gradients Network::computeGradients(const std::vector<double>& inputData, const std::vector<double>& target) {
	Gradients gradients;
	gradients.weightGradients.resize(layers.size());
	gradients.biasGradients.resize(layers.size());

	std::vector<double> output = this->frontpropogate(inputData);
	std::vector<double> gradients_output = derivativeCost_Output(target, output);

	Layer* outputLayer = layers.back();
	std::vector<double> zValues = outputLayer->getZValues();

	// Compute delta for output layer
	std::vector<double> delta_output(zValues.size());
	for (int i = 0; i < zValues.size(); i++) {
		delta_output[i] = gradients_output[i] * derivativeActivation_Z(zValues[i]);
	}

	std::vector<std::vector<double>> deltas(layers.size());
	deltas.back() = delta_output;

	// Backpropagate deltas
	for (int l = layers.size() - 2; l >= 0; l--) {
		Layer* currentLayer = layers[l];
		Layer* nextLayer = layers[l + 1];
		deltas[l].resize(currentLayer->getNeurons().size());

		for (int i = 0; i < currentLayer->getNeurons().size(); i++) {
			double sum = 0.0;
			for (int j = 0; j < nextLayer->getNeurons().size(); j++) {
				sum += nextLayer->getNeurons()[j]->getWeightAtIndex(i) * deltas[l + 1][j];
			}
			deltas[l][i] = sum * derivativeActivation_Z(currentLayer->getZValues()[i]);
		}
	}

	// Accumulate weight and bias gradients
	std::vector<double> prevActivations = inputData;
	for (int l = 0; l < layers.size(); l++) {
		Layer* layer = layers[l];
		gradients.weightGradients[l].resize(layer->getNeurons().size());
		gradients.biasGradients[l].resize(layer->getNeurons().size());

		for (int n = 0; n < layer->getNeurons().size(); n++) {
			Neuron* neuron = layer->getNeurons()[n];
			gradients.biasGradients[l][n] = deltas[l][n];

			for (int w_i = 0; w_i < prevActivations.size(); w_i++) {
				gradients.weightGradients[l][n].push_back(deltas[l][n] * prevActivations[w_i]);
			}
		}
		prevActivations = layer->getAValues();
	}

	return gradients;
}

void Network::trainBatch(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<double>>& batchTargets, const NetworkParameters& param) {

	Gradients accumulatedGradients;

	// Initialize gradients to zero
	accumulatedGradients.weightGradients.resize(layers.size());
	accumulatedGradients.biasGradients.resize(layers.size());

	for (int l = 0; l < layers.size(); l++) {
		accumulatedGradients.biasGradients[l].resize(layers[l]->getNeurons().size(), 0.0);
		accumulatedGradients.weightGradients[l].resize(layers[l]->getNeurons().size());
	}

	// Accumulate gradients for the entire batch
	for (int j = 0; j < batchInputs.size(); j++) {
		Gradients gradients = computeGradients(batchInputs[j], batchTargets[j]);

		for (int l = 0; l < layers.size(); l++) {
			for (int n = 0; n < layers[l]->getNeurons().size(); n++) {
				accumulatedGradients.biasGradients[l][n] += gradients.biasGradients[l][n];

				accumulatedGradients.weightGradients[l][n].resize(gradients.weightGradients[l][n].size(), 0.0);
				for (int w = 0; w < gradients.weightGradients[l][n].size(); w++) {
					accumulatedGradients.weightGradients[l][n][w] += gradients.weightGradients[l][n][w];
				}
			}
		}
	}

	// Apply averaged gradients using ADAM optimizer
	updateWeightsAndBiases(accumulatedGradients, param);

}

void Network::train(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& targets, const NetworkParameters& param) {

	int dataSize = inputData.size();
	std::vector<int> indices(dataSize);
	for (int i = 0; i < dataSize; i++) {
		indices[i] = i;
	}

	for (int e = 0; e < param.epochs; e++) {

		// Fisher-Yates shuffle
		for (int i = dataSize - 1; i > 0; i--) {
			int j = rand() % (i + 1);
			std::swap(indices[i], indices[j]);
		}

		for (int start = 0; start < dataSize; start += param.batchSize) {
			int end = std::min(start + param.batchSize, dataSize);

			std::vector<std::vector<double>> batchInputs;
			std::vector<std::vector<double>> batchTargets;

			batchInputs.resize(end - start);
			batchTargets.resize(end - start);

			for (int i = start; i < end; i++) {
				batchInputs[i - start] = inputData[indices[i]];
				batchTargets[i - start] = targets[indices[i]];
			}

			trainBatch(batchInputs, batchTargets, param);
		}
	}
}

void Network::updateWeightsAndBiases(const Gradients& gradients, const NetworkParameters& param) {

	t++;

	for (int l = 0; l < layers.size(); l++) {
		Layer* layer = layers[l];

		mWeights[l].resize(layer->getNeurons().size());
		vWeights[l].resize(layer->getNeurons().size());
		mBiases[l].resize(layer->getNeurons().size(), 0.0);
		vBiases[l].resize(layer->getNeurons().size(), 0.0);

		for (int n = 0; n < layer->getNeurons().size(); n++) {
			Neuron* neuron = layer->getNeurons()[n];

			mWeights[l][n].resize(neuron->getWeights().size(), 0.0);
			vWeights[l][n].resize(neuron->getWeights().size(), 0.0);

			// update bias
			double gradBias = gradients.biasGradients[l][n];
			mBiases[l][n] = param.beta1 * mBiases[l][n] + (1 - param.beta1) * gradBias;
			vBiases[l][n] = param.beta2 * vBiases[l][n] + (1 - param.beta2) * (gradBias * gradBias);

			double mHatBias = mBiases[l][n] / (1 - pow(param.beta1, t));
			double vHatBias = vBiases[l][n] / (1 - pow(param.beta2, t));

			double updatedBias = neuron->getBias() - param.learningRate * mHatBias / (sqrt(vHatBias) + param.epsilon);
			neuron->setBias(updatedBias);

			// update weights
			std::vector<double> weights = neuron->getWeights();
			for (int w = 0; w < weights.size(); w++) {
				double gradWeight = gradients.weightGradients[l][n][w];
				mWeights[l][n][w] = param.beta1 * mWeights[l][n][w] + (1 - param.beta1) * gradWeight;
				vWeights[l][n][w] = param.beta2 * vWeights[l][n][w] + (1 - param.beta2) * (gradWeight * gradWeight);

				double mHatWeight = mWeights[l][n][w] / (1 - pow(param.beta1, t));
				double vHatWeight = vWeights[l][n][w] / (1 - pow(param.beta2, t));

				weights[w] -= param.learningRate * mHatWeight / (sqrt(vHatWeight) + param.epsilon);
			}

			neuron->setWeights(weights);
		}
	}
}