#pragma once

#ifndef ActivationFunction_H
#define ActivationFunction_H

#include <cmath>

class ActivationFunction {
public:
	ActivationFunction();

    virtual double activate(double x) = 0;
    virtual double derivative(double x) = 0;
    virtual ~ActivationFunction() {}
};

class ReLU : public ActivationFunction {
public:
	ReLU();
	double activate(double x) override;
	double derivative(double x) override;
};

class LeakyReLU : public ActivationFunction {
private:
	double alpha;
public:
	LeakyReLU();
	LeakyReLU(double alpha);
	double activate(double x) override;
	double derivative(double x) override;
};

class Sigmoid : public ActivationFunction {
public:
	Sigmoid();
	double activate(double x) override;
	double derivative(double x) override;
};

class Tanh : public ActivationFunction {
public:
	Tanh();
	double activate(double x) override;
	double derivative(double x) override;
};

#endif // !ActivationFunction_H