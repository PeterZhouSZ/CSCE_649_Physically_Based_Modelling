#pragma once
#ifndef __Cloth__
#define __Cloth__

#include <vector>
#include <memory>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>
#include <Eigen/Sparse>

class Particle;
class Spring;
class MatrixStack;
class Program;

class Cloth
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	Cloth(int rows, int cols,
		  const Eigen::Vector3d &x00,
		  const Eigen::Vector3d &x01,
		  const Eigen::Vector3d &x10,
		  const Eigen::Vector3d &x11,
		  double mass,
		  double stiffness,
		  const Eigen::Vector2d &damping);
	virtual ~Cloth();
	
	void tare();
	void reset();
	void updatePosNor();
	void step(double h, const Eigen::Vector3d &grav, const std::vector< std::shared_ptr<Particle> > spheres);
	Eigen::MatrixXd runge4(Eigen::MatrixXd state, double h, Eigen::VectorXd f, Eigen::MatrixXd M);
	Eigen::MatrixXd derivative(Eigen::MatrixXd state, double h, Eigen::VectorXd f, Eigen::MatrixXd M);
	void init();
	void draw(std::shared_ptr<MatrixStack> MV, const std::shared_ptr<Program> p) const;
	
private:
	int rows;
	int cols;
	int n;
	double mm;
	Eigen::Vector2d damping;
	std::vector< std::shared_ptr<Particle> > particles;
	std::vector< std::shared_ptr<Spring> > springs;
	
	Eigen::VectorXd pos;
	Eigen::VectorXd v;
	Eigen::VectorXd f;
	Eigen::MatrixXd M;
	Eigen::MatrixXd K;
	Eigen::MatrixXd state;
	
	std::vector<unsigned int> eleBuf;
	std::vector<float> posBuf;
	std::vector<float> norBuf;
	std::vector<float> texBuf;
	unsigned eleBufID;
	unsigned posBufID;
	unsigned norBufID;
	unsigned texBufID;
};

#endif
