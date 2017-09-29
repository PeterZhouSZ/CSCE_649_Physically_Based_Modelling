#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Particle;
class Spring;
class MatrixStack;
class Program;

class Box
{
public:
	Box(double f, double b, double r, double l, double u, double d);
	~Box();

	double forward; // + z
	double back; // - z
	double right; // + x
	double left; // - x
	double up; // + y
	double down; // - y

};

