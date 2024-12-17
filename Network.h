#pragma once

#ifndef Network_H
#define Network_H

#include <vector>
#include "Layer.h"

class Network {
private:
	std::vector<Layer*> layers;

	double costFunction(const std::vector<double>& target, const std::vector<double>& output);
	double derivativeCost_Output(double target, double output);

	// derivative of cost function with respect to output (dC/dO)
	std::vector<double> gradientCalculation(const std::vector<double>& target, const std::vector<double>& output); 
public:
	Network(std::vector<int> topology);
	std::vector<double> frontpropogate(const std::vector<double>& inputData);
	void backpropagate(const std::vector<double>& target);
};
#endif // !Network_H