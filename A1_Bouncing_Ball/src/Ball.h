#pragma once
#include <vector>
#include <memory>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Particle;
class Spring;
class MatrixStack;
class Program;
class Box;

class Ball
{
public:
	Ball(
		const Eigen::Vector3d &center,
		const Eigen::Vector3d &velocity,
		double m,
		double radius
	);

	virtual ~Ball();
	void step_poly(double h, const Eigen::Vector3d &grav, const std::shared_ptr<Box> box);
	void step_w(double h, const Eigen::Vector3d &grav, const std::shared_ptr<Box> box);

	void step(double h, const Eigen::Vector3d &grav, const std::shared_ptr<Box> box);
	void init();
	void tare();
	void reset();
	void draw(std::shared_ptr<MatrixStack> MV, const std::shared_ptr<Program> p, std::shared_ptr<Box> box)const;
	void updatePosNor();

private:
	int n;
    double r;
	double m;
	int rings;
	int sectors;

	Eigen::Vector3d v;
	Eigen::Vector3d center;
	Eigen::Vector3d center_0; // initial position
	Eigen::Vector3d v0; // initial velocity



	std::vector<unsigned int> eleBuf;
	std::vector<float> posBuf;
	std::vector<float> norBuf;
	std::vector<float> texBuf;

	unsigned eleBufID;
	unsigned posBufID;
	unsigned norBufID;
	unsigned texBufID;
};