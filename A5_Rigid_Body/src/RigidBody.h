#pragma once
#ifndef __RigidBody__
#define __RigidBody__

#include <vector>
#include <memory>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <tetgen.h>

class Particle;
class MatrixStack;
class Program;

struct RBState	
{
	// constants
	double mass;
	Eigen::Matrix3d Ibody;
	Eigen::Matrix3d Ibodyinv;

	// states
	Eigen::Vector3d x;
	Eigen::Matrix3d R;
	Eigen::Vector3d P;
	Eigen::Vector3d L;

	// derived
	Eigen::Matrix3d Iinv;
	Eigen::Vector3d v;
	Eigen::Vector3d omega;

	// computed
	Eigen::Vector3d force;
	Eigen::Vector3d torque;

	RBState()
	{
		mass = 10.0;
		double i = 1.0 / 12.0 * (2.0*2.0 + 2.0*2.0)* mass;
		Ibody.setIdentity();
		Ibody = Ibody * i;
		Ibodyinv=Ibody.inverse();
		x = Eigen::Vector3d(0.0, 0.0, 0.0);
		v = Eigen::Vector3d(0.0, 0.0, 0.0);
		omega = Eigen::Vector3d(1.0, 0.0, 0.0);	
		R.setIdentity();
		P = mass * v;
		L = Ibody * omega;

		Iinv.setZero();	
		force.setZero();
		torque.setZero();
	}
};

class RigidBody {
	public:
		
		RigidBody(int id, int rot, int type, Eigen::Vector3d pos, Eigen::Vector3d v0, Eigen::Vector3d omega0);
		void init();
		void setVelocity(Eigen::Vector3d v);
		void setOmega(Eigen::Vector3d omega);
		void setPosition(Eigen::Vector3d pos);
		
		// Basic 
		void state2array(std::shared_ptr<RBState> rb, Eigen::VectorXd &y);
		void array2state(std::shared_ptr<RBState> rb, Eigen::VectorXd y);
		void array2bodies(Eigen::VectorXd x);
		void bodies2array(Eigen::VectorXd &x);
		void computeForceTorque(double t, std::shared_ptr<RBState> rb);
		void Dxdt(double t, Eigen::VectorXd x, Eigen::VectorXd &xdot);
		void ddtstate2array(std::shared_ptr<RBState> rb, Eigen::VectorXd &xdot);	
		void step(double h, const Eigen::Vector3d &grav, std::vector< std::shared_ptr<RigidBody> >rigidbodies, int checkedge);
		void updatePosNor();

		// Draw
		void draw(std::shared_ptr<MatrixStack> MV, const std::shared_ptr<Program> p)const;
		void drawAABB()const;

		// Helper
		Eigen::Matrix3d Star(Eigen::Vector3d a);
		double randDouble(double l, double h);
		void normalizeR();
		bool rayTriangleIntersects(Eigen::Vector3d v1, Eigen::Vector3d v2, Eigen::Vector3d v3, Eigen::Vector3d dir, Eigen::Vector3d pos);

		// Collision
		void computeAABB();
		void collideBox(Eigen::VectorXd box);
		void collideRBs(const std::vector< std::shared_ptr<RigidBody> >rigidbodies, int checkedge);
		
		virtual ~RigidBody();
		
		int id;
		double cr;
		int nbodies;
		int nTriFaces;
		int nEdges;
		int nVerts;
		tetgenio in, out;

		std::vector < std::shared_ptr<Particle> > nodes;
		std::vector< std::shared_ptr<RBState> > bodies;
		Eigen::Vector3d min3, max3, index_min3, index_max3;
		Eigen::MatrixXd normal;

private:
	double t0;
	double t;
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