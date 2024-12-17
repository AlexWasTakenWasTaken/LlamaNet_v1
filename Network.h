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

};
#endif // !Network_H