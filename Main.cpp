#include <iostream>
#include <chrono>

#include "Neuron.h"
#include "Layer.h"
#include "Network.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

void loadDataset(const std::string& fileName,
    std::vector<std::vector<double>>& inputs,
    std::vector<std::vector<double>>& targets) {
    std::ifstream file(fileName);

    std::string line;
	int i = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> rowValues;

        while (std::getline(ss, cell, ',')) {
            rowValues.push_back(std::stod(cell));
        }

        int label = static_cast<int>(rowValues[0]);

        std::vector<double> inputValues(rowValues.begin() + 1, rowValues.end());
        for (auto& val : inputValues) {
            val = val / 255.0;
        }

        std::vector<double> targetValues(10, 0.0);
        if (label >= 0 && label < 10) {
            targetValues[label] = 1.0;
        }

        inputs.push_back(inputValues);
        targets.push_back(targetValues);
    }

    file.close();
}

int main() {
	std::srand(static_cast<unsigned>(std::time(0)));



	std::vector<int> topology = { 784, 128, 64, 10 };

	ActivationFunction* activationFunction = new LeakyReLU(0.01);
	Network* network = new Network(topology, activationFunction); 
	
	Network::NetworkParameters networkParameter;
	networkParameter.batchSize = 30;

    std::vector<std::vector<double>> trainingInputs;
    std::vector<std::vector<double>> trainingTargets;
    loadDataset("mnist_train.csv", trainingInputs, trainingTargets);

    std::vector<std::vector<double>> testingInputs;
    std::vector<std::vector<double>> testingTargets;
    loadDataset("mnist_test.csv", testingInputs, testingTargets);


    const auto startTime = std::chrono::steady_clock::now();
    for (int i = 0; i < 25; i++) {

        network->train(trainingInputs, trainingTargets, networkParameter);

        auto endTime = std::chrono::steady_clock::now();


        int correctCount = 0;
        for (int i = 0; i < (int)testingInputs.size(); ++i) {
            std::vector<double> output = network->frontpropogate(testingInputs[i]);

            // Find the predicted label by getting the index of the max element in output
            int predictedLabel = (int)std::distance(output.begin(), std::max_element(output.begin(), output.end()));

            // Find the actual label by getting the index of the max element in the target
            int actualLabel = (int)std::distance(testingTargets[i].begin(), std::max_element(testingTargets[i].begin(), testingTargets[i].end()));

            if (predictedLabel == actualLabel) {
                correctCount++;
            }
        }

        double accuracy = (static_cast<double>(correctCount) / testingInputs.size()) * 100.0;
        std::cout << "Accuracy: " << accuracy << "%, Epoch: " << i << std::endl;

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
    }

    return 0;
}