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
	// derivative of Leaky ReLU activation function
	return (z > 0) ? 1.0 : 0.01;
}

std::vector<double> Network::derivativeCost_Output(const std::vector<double>& target, const std::vector<double>& output) {
	std::vector<double> gradients(output.size());

	for (int i = 0; i < target.size(); i++) {
		gradients[i] = costFunctionDerivative(target[i], output[i]);
	}
	return gradients;
}

Network::Network(std::vector<int> topology) {
	for (int i = 1; i < topology.size(); i++) {
		layers.push_back(new Layer(topology[i], topology[i - 1]));
	}
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

void Network::updateWeightsAndBiases(const Gradients& gradients, double learningRate, int batchSize) {
    for (int l = 0; l < layers.size(); l++) {
        Layer* layer = layers[l];
        for (int n = 0; n < layer->getNeurons().size(); n++) {
            Neuron* neuron = layer->getNeurons()[n];

            double updatedBias = neuron->getBias() - (learningRate / batchSize) * gradients.biasGradients[l][n];
            neuron->setBias(updatedBias);

            std::vector<double> weights = neuron->getWeights();
            for (int w = 0; w < weights.size(); w++) {
                weights[w] -= (learningRate / batchSize) * gradients.weightGradients[l][n][w];
            }
            neuron->setWeights(weights);
        }
    }
}

void Network::trainBatch(const std::vector<std::vector<double>>& batchInputs, const std::vector<std::vector<double>>& batchTargets, int epochs, double learningRate) {
    for (int i = 0; i < epochs; i++) {
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
                        /*if (accumulatedGradients.weightGradients[l][n].size() <= w) {
                            accumulatedGradients.weightGradients[l][n].push_back(0.0);
                        }*/
                        accumulatedGradients.weightGradients[l][n][w] += gradients.weightGradients[l][n][w];
                    }
                }
            }
        }

        // Apply averaged gradients
        updateWeightsAndBiases(accumulatedGradients, learningRate, batchInputs.size());
    }
}