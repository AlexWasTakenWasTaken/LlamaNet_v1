#pragma once

#ifndef Network_H
#define Network_H

#include <vector>
#include "Layer.h"

class Network {
private:
	std::vector<Layer*> layers;
public:
	Network(std::vector<int> topology);
	std::vector<double> predict(const std::vector<double>& inputData);

	double costFunction(const std::vector<double>& target, const std::vector<double>& output);
	double gradientCalculation(const std::vector<double>& target, const std::vector<double>& output);
};
#endif // !Network_H