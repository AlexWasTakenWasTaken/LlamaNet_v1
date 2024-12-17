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

void Network::backpropogate(const std::vector<double>& inputData, const std::vector<double>& target) {
	
	// default learning rate: 0.01
    const double learningRate = 0.01;

    std::vector<double> output = this->frontpropogate(inputData);
    std::vector<double> gradients = derivativeCost_Output(target, output);

    Layer* outputLayer = layers.back();

    std::vector<double> zValues = outputLayer->getZValues();
    std::vector<double> aValues = outputLayer->getAValues();

	// output layer delta calculation
    // delta for output layer: delta = dCost/dOutput * dActivation/dZ
    std::vector<double> delta_output(zValues.size());
    for (int i = 0; i < zValues.size(); i++) {

        double dA_dZ = derivativeActivation_Z(zValues[i]);
        delta_output[i] = gradients[i] * dA_dZ;
    }

	// hidden layer delta calculation
    std::vector<std::vector<double>> layerDeltas(layers.size());
    layerDeltas.back() = delta_output;

    // delta_l = (W_{l+1}^T * delta_{l+1}) * dA/dZ_l
    for (int l = layers.size() - 2; l >= 0; l--) {
        Layer* currentLayer = layers[l];
        Layer* nextLayer = layers[l + 1];

        std::vector<double> zValues = currentLayer->getZValues();
        std::vector<double> dC_dZ(zValues.size(), 0.0);

        // dC/dZ_l = (W_{l+1}^T * delta_{l+1}) element-wise multiplied by dA/dZ_l
        for (int i = 0; i < currentLayer->getNumNeurons(); i++) {
            double sum = 0.0;
            // sum over j: W_{l+1}(i,j)*delta_{l+1}(j)
            for (int j = 0; j < nextLayer->getNumNeurons(); j++) {
                double w = nextLayer->getNeuronAtIndex(j)->getWeightAtIndex(i);
                sum += w * layerDeltas[l + 1][j];
            }

            double dA_dZ = derivativeActivation_Z(zValues[i]);
            
            dC_dZ[i] = sum * dA_dZ;
        }

        layerDeltas[l] = dC_dZ;
    } 

    // adjust weights and biases
    std::vector<double> prevActivations = inputData;
    for (int l = 0; l < (int)layers.size(); l++) {
        Layer* layer = layers[l];

        // current layer deltas = layerDeltas[l]
        std::vector<double> deltas = layerDeltas[l];

        for (int n = 0; n < layer->getNumNeurons(); n++) {
            Neuron* neuron = layer->getNeuronAtIndex(n);

            double newBias = neuron->getBias() - learningRate * deltas[n];
            neuron->setBias(newBias);

            std::vector<double> w = neuron->getWeights();
            for (int w_i = 0; w_i < (int)w.size(); w_i++) {
                double newWeight = w[w_i] - learningRate * deltas[n] * prevActivations[w_i];
                w[w_i] = newWeight;
            }
            neuron->setWeights(w);
        }

        prevActivations = layer->getAValues();
    }
}