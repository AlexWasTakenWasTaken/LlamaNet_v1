#include "ActivationFunction.h"

ActivationFunction::ActivationFunction() {}

ReLU::ReLU() {}
double ReLU::activate(double x) {
	return (x > 0 ? x : 0);
}
double ReLU::derivative(double x) {
	return (x > 0 ? 1 : 0);
}

LeakyReLU::LeakyReLU() {
	alpha = 0.01;
}
LeakyReLU::LeakyReLU(double alpha) {
	this->alpha = alpha;
}
double LeakyReLU::activate(double x) {
	return (x > 0 ? x : alpha * x);
}
double LeakyReLU::derivative(double x) {
	return (x > 0 ? 1 : alpha);
}

Sigmoid::Sigmoid() {}
double Sigmoid::activate(double x) {
	return 1 / (1 + exp(-x));
}
double Sigmoid::derivative(double x) {
	return x * (1 - x);
}

Tanh::Tanh() {}
double Tanh::activate(double x) {
	return tanh(x);
}
double Tanh::derivative(double x) {
	return 1 - x * x;
}