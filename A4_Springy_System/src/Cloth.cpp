#include <iostream>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Cloth.h"
#include "Particle.h"
#include "Spring.h"
#include "MatrixStack.h"
#include "Program.h"
#include "GLSL.h"
#include <stdio.h>
#include <igl/mosek/mosek_quadprog.h>

#define RK4 2
#define EULER 1
#define IMPLICIT 3

// PIN : 1  fix one top points
//       2  fix two top points
//       0  fix no points

#define DEMO 1
#define PIN 2
#define MODE 2

// Demo1a: Euler
//#define DEMO 1
//#define PIN 2
//#define MODE 1

// Demo1b: RK4
//#define DEMO 1
//#define PIN 2
//#define MODE 2

// Demo2: Collision with sphere
//#define PIN 2
//#define DEMO 2
//#define MODE 3

// Demo3: Collision with face
//#define DEMO 3
//#define PIN 0
//#define MODE 3

// Demo4: Edge to edge collisions
//#define DEMO 4
//#define PIN 0
//#define MODE 3

using namespace std;
using namespace Eigen;
typedef Eigen::Triplet<double> T;

// This test function is adapted from Moller-Trumbore intersection algorithm. 
// See https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm 

bool rayTriangleIntersects(Vector3d v1, Vector3d v2, Vector3d v3, Vector3d dir, Vector3d pos) {

	Vector3d e1 = v2 - v1;
	Vector3d e2 = v3 - v1;

	// Calculate planes normal vector
	//cross product
	Vector3d pvec = dir.cross(e2);

	//dot product
	double det = e1.dot(pvec);

	// Ray is parallel to plane
	if (det <1e-8 && det > -1e-8) {
		return false;
	}

	double inv_det = 1 / det;

	// Distance from v1 to ray pos
	Vector3d tvec = pos - v1;
	double u = (tvec.dot(pvec))*inv_det;
	if (u < 0 || u > 1) {
		return false;
	}

	Vector3d qvec = tvec.cross(e1);
	double v = dir.dot(qvec) * inv_det;
	if (v<0 || u + v>1) {
		return false;
	}

	double t = e2.dot(qvec) * inv_det;
	if (t > 1e-8) return true;
	return false;
}


shared_ptr<Spring> createSpring(const shared_ptr<Particle> p0, const shared_ptr<Particle> p1, double E)
{
	auto s = make_shared<Spring>(p0, p1);
	s->E = E;
	Vector3d x0 = p0->x;
	Vector3d x1 = p1->x;
	Vector3d dx = x1 - x0;
	s->L = dx.norm();
	return s;
}

Cloth::Cloth(int rows, int cols,
			 const Vector3d &x00,
			 const Vector3d &x01,
			 const Vector3d &x10,
			 const Vector3d &x11,
			 double mass,
			 double stiffness,
			 const Vector2d &damping)
{
	assert(rows > 1);
	assert(cols > 1);
	assert(mass > 0.0);
	assert(stiffness > 0.0);
	
	this->rows = rows;
	this->cols = cols;
	this->damping = damping;
	
	// Create particles
	n = 0;
	double r = 0.02; // Used for collisions
	int nVerts = rows*cols;
	 mm = mass / nVerts;
	for(int i = 0; i < rows; ++i) {
		double u = i / (rows - 1.0);
		Vector3d x0 = (1 - u)*x00 + u*x10;
		Vector3d x1 = (1 - u)*x01 + u*x11;
		for(int j = 0; j < cols; ++j) {
			double v = j / (cols - 1.0);
			Vector3d x = (1 - v)*x0 + v*x1;
			auto p = make_shared<Particle>();
			particles.push_back(p);
			p->r = r;
			p->x = x;
			p -> x_old = x;
			p->v << 0.0, 0.0, 0.0;
			p->v_old = p->v;
			p->m = mass/(nVerts);
			// Pin two particles
			if(PIN == 2){
				if (i == 0 && (j == 0 || j == cols - 1)) {
					p->fixed = true;
					p->i = -1;
				}
				else {
					p->fixed = false;
					p->i = n;
					n += 3;
				}

			}
			if (PIN == 1) {
				if (i == 0 && ( j == cols - 1)) {
					p->fixed = true;
					p->i = -1;
				}
				else {
					p->fixed = false;
					p->i = n;
					n += 3;
				}
			}
			if (PIN == 0) {
					p->fixed = false;
					p->i = n;
					n += 3;
			}	
		}
	}
	
	// Create x springs
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols-1; ++j) {
			int k0 = i*cols + j;
			int k1 = k0 + 1;
			springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		}
	}
	
	// Create y springs
	for(int j = 0; j < cols; ++j) {
		for(int i = 0; i < rows-1; ++i) {
			int k0 = i*cols + j;
			int k1 = k0 + cols;
			springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		}
	}
	
	// Create shear springs
	for(int i = 0; i < rows-1; ++i) {
		for(int j = 0; j < cols-1; ++j) {
			int k00 = i*cols + j;
			int k10 = k00 + 1;
			int k01 = k00 + cols;
			int k11 = k01 + 1;
			springs.push_back(createSpring(particles[k00], particles[k11], stiffness));
			springs.push_back(createSpring(particles[k10], particles[k01], stiffness));
		}
	}

	// Build system matrices and vectors
	M.resize(n,n);
	K.resize(n,n);
	v.resize(n);
	f.resize(n);
	pos.resize(n);
	state.resize(n, 2);

	// Build vertex buffers
	posBuf.clear();
	norBuf.clear();
	texBuf.clear();
	eleBuf.clear();
	posBuf.resize(nVerts*3);
	norBuf.resize(nVerts*3);
	updatePosNor();
	// Texture coordinates (don't change)
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			texBuf.push_back(i/(rows-1.0));
			texBuf.push_back(j/(cols-1.0));
		}
	}
	// Elements (don't change)
	for(int i = 0; i < rows-1; ++i) {
		for(int j = 0; j < cols; ++j) {
			int k0 = i*cols + j;
			int k1 = k0 + cols;
			// Triangle strip
			eleBuf.push_back(k0);
			eleBuf.push_back(k1);
		}
	}
}

Cloth::~Cloth()
{
}

void Cloth::tare()
{
	for(int k = 0; k < (int)particles.size(); ++k) {
		particles[k]->tare();
	}
}

void Cloth::reset()
{
	for(int k = 0; k < (int)particles.size(); ++k) {
		particles[k]->reset();
	}
	updatePosNor();
}

void Cloth::updatePosNor()
{
	// Position
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			int k = i*cols + j;
			Vector3d x = particles[k]->x;
			posBuf[3*k+0] = x(0);
			posBuf[3*k+1] = x(1);
			posBuf[3*k+2] = x(2);
		}
	}
	// Normal
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			// Each particle has four neighbors
			//
			//      v1
			//     /|\
			// u0 /_|_\ u1
			//    \ | /
			//     \|/
			//      v0
			//
			// Use these four triangles to compute the normal
			int k = i*cols + j;
			int ku0 = k - 1;
			int ku1 = k + 1;
			int kv0 = k - cols;
			int kv1 = k + cols;
			Vector3d x = particles[k]->x;
			Vector3d xu0, xu1, xv0, xv1, dx0, dx1, c;
			Vector3d nor(0.0, 0.0, 0.0);
			int count = 0;
			// Top-right triangle
			if(j != cols-1 && i != rows-1) {
				xu1 = particles[ku1]->x;
				xv1 = particles[kv1]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Top-left triangle
			if(j != 0 && i != rows-1) {
				xu1 = particles[kv1]->x;
				xv1 = particles[ku0]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Bottom-left triangle
			if(j != 0 && i != 0) {
				xu1 = particles[ku0]->x;
				xv1 = particles[kv0]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Bottom-right triangle
			if(j != cols-1 && i != 0) {
				xu1 = particles[kv0]->x;
				xv1 = particles[ku1]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			nor /= count;
			nor.normalize();
			norBuf[3*k+0] = nor(0);
			norBuf[3*k+1] = nor(1);
			norBuf[3*k+2] = nor(2);
		}
	}
}

MatrixXd Cloth::derivative(MatrixXd state, double h, VectorXd f, MatrixXd M) {
	VectorXd x0 = state.block(0, 0, n, 1);
	VectorXd v0 = state.block(0, 1, n, 1);
	VectorXd a = f;
	MatrixXd state1(n,2);
	state1.block(0, 0, n, 1) = v0;
	state1.block(0, 1, n, 1) = a;
	return state1;
}

MatrixXd Cloth::runge4(MatrixXd state, double h, VectorXd f, MatrixXd M) {
	MatrixXd K1, K2, K3, K4, K5;
	K1.resize(n, 2);
	K2.resize(n, 2);
	K3.resize(n, 2);
	K4.resize(n, 2);
	K5.resize(n, 2);

	K1 = h * derivative(state, h, f, M);

	K2 =  h * derivative(state + 1.0 / 2.0 * K1, h/2.0, f, M);

	K3 =  h * derivative(state + 1.0 / 2.0 * K2, h/2.0, f, M);

	K4 = h * derivative(state + K3, h, f, M);
	K5 = state + 1.0 / 6.0 * (K1 + 2.0 * K2 + 2.0 * K3 + K4);
	return K5;
}

void Cloth::step(double h, const Vector3d &grav, const vector< shared_ptr<Particle> > spheres)
{
	M.setZero();
	K.setZero();
	v.setZero();
	f.setZero();
	pos.setZero();
	state.setZero();

	// This mode is for explicit method: Euler and RK4
	if (MODE != 3) {
		for (int p = 0; p < particles.size(); p++) {
			int index = particles[p]->i;
			if(index != -1){
				double mass = particles[p]->m;
				Matrix3d A;
				A.setIdentity();
				A *= mass;
				M.block<3,3>(index, index) = A;
				pos.segment<3>(index) = particles[p]->x;
				v.segment<3>(index) = particles[p]->v;
				f.segment<3>(index) = mass * grav;
			}
		}

		for (int i = 0; i < springs.size(); i++){
			int idx0 = springs[i]->p0->i;
			int idx1 = springs[i]->p1->i;
			Vector3d dx = springs[i]->p1->x - springs[i]->p0->x;

			// fs
			double l = sqrt(pow(dx(0),2)+pow(dx(1),2)+pow(dx(2),2));
			Vector3d fs = (springs[i]->E)*(l - springs[i]->L)/l * dx;

			int ei = 0;
			int si = 0;
			if(idx0 > idx1){
			 	ei = idx0;
			 	si = idx1;
			}else{
			 	ei = idx1;
			 	si = idx0;
			}

			if(idx0 != -1){
			 	f.segment<3>(idx0) += fs;
			}

			if(idx1 != -1){
			 	f.segment<3>(idx1) -= fs;
			}

		}

		if (MODE == 1) {
			state.block(0, 0, n, 1) = pos;
			state.block(0, 1, n, 1) = v;
			//VectorXd x = (M).ldlt().solve(M*v + h*f);
			f = f / mm;
			MatrixXd state_new(n, 2);
			state_new = state + h * derivative(state, h, f, M);
			VectorXd vv = state_new.block(0, 1, n, 1);
			VectorXd xx = state_new.block(0, 0, n, 1);
			for (int i = 0; i < particles.size(); i++) {
				if (particles[i]->i != -1) {
					particles[i]->v = vv.segment<3>(particles[i]->i);
				}
			}

			for (int i = 0; i < particles.size(); i++) {
				if (particles[i]->i != -1) {
					//particles[i]->x += particles[i]->v * h;
					particles[i]->x = xx.segment<3>(particles[i]->i);
				}
			}
			updatePosNor();
		}

		if (MODE == 2) {
			state.block(0, 0, n, 1) = pos;
			state.block(0, 1, n, 1) = v;
			MatrixXd state_new(n, 2);
			state_new.setZero();
			f = f / mm;
			state_new = runge4(state, h, f, M);
			
			VectorXd vv = state_new.block(0, 1, n, 1);
			VectorXd xx = state_new.block(0, 0, n, 1);
			
			for (int i = 0; i < particles.size(); i++) {
				if (particles[i]->i != -1) {
					particles[i]->v = vv.segment<3>(particles[i]->i);
				}
			}

			for (int i = 0; i < particles.size(); i++) {
				if (particles[i]->i != -1) {
					particles[i]->x = xx.segment<3>(particles[i]->i);
				}
			}
			updatePosNor();
		}
	}

	// Sparse Version
	// The mode is for Implicit integration

	// Construct matrix A and vector b while looping through particles and springs
	if (MODE == 3) {
		vector<T> A_;
		Eigen::VectorXd b;
		b.resize(n);
		b.setZero();

		for (int i = 0; i<particles.size(); i++) {
			int idx = particles[i]->i;
			if (idx != -1) {
				double mass = particles[i]->m;
				Vector3d vel = particles[i]->v;

				// filling vector v for initial guess
				v.segment<3>(idx) = vel;

				for (int j = 0; j<3; j++) {
					A_.push_back(T(idx + j, idx + j, mass));
					b(idx + j) += mass * vel(j) + h * grav(j) * mass;

				}
			}
		}

		for (int i = 0; i<springs.size(); i++) {

			int idx0 = springs[i]->p0->i;
			int idx1 = springs[i]->p1->i;

			Vector3d dx = springs[i]->p1->x - springs[i]->p0->x;

			// The vector Fs for one spring
			double l = sqrt(pow(dx(0), 2) + pow(dx(1), 2) + pow(dx(2), 2));
			Vector3d fs = (springs[i]->E)*(l - springs[i]->L) / l * dx;

			if (idx0 != -1) {
				for (int t = 0; t<3; t++) {
					b(idx0 + t) += h * fs(t);
				}
			}

			if (idx1 != -1) {
				for (int t = 0; t<3; t++) {
					b(idx1 + t) -= h * fs(t);
				}
			}
			// The matrix Ks for one spring

			Matrix3d xxT = dx * dx.transpose();
			double xTx = dx.transpose() * dx;
			Matrix3d I = Matrix3d::Identity();
			Matrix3d ks = springs[i]->E / pow(l, 2) * ((1 - (l - springs[i]->L) / l)*xxT + (l - springs[i]->L) / l * xTx * I);

			if (idx0 != -1 && idx1 != -1) {

				for (int k = 0; k < 3; k++) {
					for (int g = 0; g < 3; g++) {
						double val = damping(1) * pow(h, 2) * ks(k, g);

						// filling the diagonal values of K
						A_.push_back(T(idx0 + k, idx0 + g, val));
						A_.push_back(T(idx1 + k, idx1 + g, val));

						// filling the negated, off-diagonal values of K
						A_.push_back(T(idx0 + k, idx1 + g, -val));
						A_.push_back(T(idx1 + k, idx0 + g, -val));
					}
				}
			}
		}
		Eigen::SparseMatrix<double> A(n, n);
		A.setFromTriplets(A_.begin(), A_.end());

		// Collision Detection and Response

		std::vector<int> index_cols;   // used for the constraint matrix
		std::vector<double> vs_lower;  // used for the velocity lower bound

		vector<T> C_; // for constraints

		int row = 0;
		auto s = spheres.front();
		
		// With the moving sphere
		for (int i = 0; i < spheres.size(); i++) {
			shared_ptr<Particle> p_s = spheres[i];
			for (int j = 0; j < particles.size(); j++) {

				shared_ptr<Particle> p = particles[j];
				Vector3d dx = p->x - p_s->x;
				Vector3d dx_n = dx.normalized();

				// Collisions with the sphere
				if (dx.norm() <= p->r + p_s->r) {
					p->isCol = true;
					p->x = ((p_s->r + p->r) * dx_n + p_s->x);
					Vector3d v_normal = -(p->v.dot(dx_n) * dx).normalized(); // save the normal of each particle that has a collision event
					p->normal = v_normal;

					// filling the constraint matrix
					for (int t = 0; t<3; t++) {
						C_.push_back(T(row, (p->i) + t, v_normal(t)));
					}

					vs_lower.push_back(s->v.dot(p->normal)); // filling the lower bound(according to the moving sphere) of velocity of each particle that has a collision 
					index_cols.push_back(p->i);
					row++;
				}
			}
		}

		// Vertex to face collision detection and response
		if (DEMO == 3) {
			int  colfaces = 1;
			// Face vertices
			Vector3d v1, v2, v3;
			v1 << 0.5, 0.5, 0.0;
			v2 << 0.0, 0.0, -0.5;
			v3 << 0.0, 0.0, 0.5;
			Vector3d normal = -(v1 - v3).cross(v2 - v3);
			double twoA = normal.norm();
			Vector3d nor = -normal / twoA;
			//cout << nor << endl;
			for (int i = 0; i < colfaces; i++) {

				for (int j = 0; j < particles.size(); j++) {

					shared_ptr<Particle> p = particles[j];
					Vector3d x_old = p->x_old;
					Vector3d x_new = p->x;
					Vector3d dir = x_new - x_old;
					double f = (x_old - v1).dot(nor) / ((x_old - v1).dot(nor) - (x_new - v1).dot(nor));
					// Collisions with the face
					if (rayTriangleIntersects(v1, v2, v3, dir, x_old) && f < 1.0) {
						p->isCol = true;
						
						Vector3d v_normal = nor; // save the normal of each particle that has a collision event
						p->normal = v_normal;
						// filling the constraint matrix
						for (int t = 0; t < 3; t++) {
							C_.push_back(T(row, (p->i) + t, v_normal(t)));
						}
						vs_lower.push_back(1); // filling the lower bound of velocity of each particle that has a collision 
						index_cols.push_back(p->i);
						row++;
					}
				}
			}

		}

		// Edge to edge collision detection and response
		if (DEMO == 4) {
			Vector3d v1, v2;
			v1 << 0.5, 0.5, 0.0;
			v2 << 0.0, 0.0, -0.5;
			
			Vector3d edge1 = v1 - v2;

			for (int i = 0; i < springs.size(); i++) {

				// a is the vector along each spring
				Vector3d a = springs[i]->p0->x - springs[i]->p1->x;

				// nor1 is the direction of paqa
				Vector3d nor1 = a.cross(edge1).normalized();

				// vector from p1 to q1
				Vector3d r = v2 - springs[i]->p0->x;

				// the value of s corresponding to spring pa is
				double s = r.dot(edge1.normalized().cross(nor1)) / a.dot(edge1.normalized().cross(nor1));

				// the value of t corresponding to edge qa is
				double t = -r.dot(a.normalized().cross(nor1)) / edge1.dot(a.normalized().cross(nor1));

				Vector3d pa = springs[i]->p0->x + s*a;
				Vector3d qa = v2 + t *edge1;
				Vector3d m = qa - pa;

				// Collision detected
				if (m.norm() < 0.01 && (s > 0 && s < 1||s==0||s==1) && (t > 0 && t < 1||t==0||t==1)) {
					
					Vector3d v_normal;
					if (nor1(0) > 0) {
						 v_normal = -nor1;
					}
					else {
						 v_normal = nor1;
					}

					// save the normal of each particle that has a collision event
					springs[i]->p0->normal = v_normal;
					springs[i]->p1->normal = v_normal;

					// filling the constraint matrix
					for (int t = 0; t < 3; t++) {
						C_.push_back(T(row, (springs[i]->p0->i) + t, v_normal(t)));	
					}
					row++;

					for (int t = 0; t < 3; t++) {
						C_.push_back(T(row, (springs[i]->p1->i) + t, v_normal(t)));
					}
					row++;

					vs_lower.push_back(1.0*(1-s)); // filling the lower bound(according to collision point) of velocity of each particle that has a collision 
					vs_lower.push_back(1.0*s);
					index_cols.push_back(springs[i]->p0->i);
					index_cols.push_back(springs[i]->p1->i);
				}
			}
		}

		int num_collisions = index_cols.size();

		if (num_collisions == 0) {
			ConjugateGradient< SparseMatrix<double> > cg;
			cg.setMaxIterations(25);
			cg.setTolerance(1e-3);
			cg.compute(A);
			VectorXd x = cg.solveWithGuess(b, v);
			for (int i = 0; i < particles.size(); i++) {
				if (particles[i]->i != -1) {
					particles[i]->v_old = particles[i]->v;
					particles[i]->v = x.segment<3>(particles[i]->i);
				}
			}

			for (int i = 0; i < particles.size(); i++) {
				if (particles[i]->i != -1) {
					particles[i]->x_old = particles[i]->x;
					particles[i]->x += particles[i]->v * h;
				}
			}
		}

		if (num_collisions > 0) {
			Eigen::SparseMatrix<double> C(num_collisions, n);
			C.setFromTriplets(C_.begin(), C_.end());
			VectorXd lc = VectorXd::Zero(num_collisions); // lc m-long vector of 0 for linear inequality
			VectorXd uc = VectorXd::Zero(num_collisions); // m-long +Inifity
			VectorXd lx;
			lx.resize(n);
			VectorXd ux;
			ux.resize(n);

			VectorXd cc = -b;

			for (int i = 0; i < n; i++) {
				lx(i) = -MSK_INFINITY;
				ux(i) = +MSK_INFINITY;
			}

			for (int i = 0; i < num_collisions; i++) {
				lc(i) = abs(vs_lower[i]);
				uc(i) = +MSK_INFINITY;
			}

			VectorXd results;
			igl::mosek::MosekData mosek_data;

			bool r = mosek_quadprog(A, cc, 0, C, lc, uc, lx, ux, mosek_data, results); // solve the qp problem

			for (int i = 0; i < particles.size(); i++) {
				if (particles[i]->i != -1) {
					particles[i]->v_old = particles[i]->v;
					particles[i]->v = results.segment<3>(particles[i]->i);
				}
			}

			for (int i = 0; i < particles.size(); i++) {
				if (particles[i]->i != -1) {
					particles[i]->x_old = particles[i]->x;
					particles[i]->x += particles[i]->v * h;
				}
			}
		}

		// Update position and normal buffers
		updatePosNor();
	}
	
}

void Cloth::init()
{
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	
	glGenBuffers(1, &norBufID);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size()*sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	
	glGenBuffers(1, &texBufID);
	glBindBuffer(GL_ARRAY_BUFFER, texBufID);
	glBufferData(GL_ARRAY_BUFFER, texBuf.size()*sizeof(float), &texBuf[0], GL_STATIC_DRAW);
	
	glGenBuffers(1, &eleBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, eleBuf.size()*sizeof(unsigned int), &eleBuf[0], GL_STATIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	
	assert(glGetError() == GL_NO_ERROR);
}

void Cloth::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> p) const
{
	// Draw mesh
	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(1.0, 0.0, 0.0).data());
	glUniform3fv(p->getUniform("kdBack"),  1, Vector3f(1.0, 1.0, 0.0).data());
	MV->pushMatrix();
	glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	int h_pos = p->getAttribute("aPos");
	glEnableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	int h_nor = p->getAttribute("aNor");
	glEnableVertexAttribArray(h_nor);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size()*sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_nor, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

	int h_tex = p->getAttribute("aTex");
	glEnableVertexAttribArray(h_tex);
	glBindBuffer(GL_ARRAY_BUFFER, texBufID);
	glBufferData(GL_ARRAY_BUFFER, texBuf.size() * sizeof(float), &texBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_tex, 2, GL_FLOAT, GL_FALSE, 0, (const void *)0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	for(int i = 0; i < rows; ++i) {
		glDrawElements(GL_TRIANGLE_STRIP, 2*cols, GL_UNSIGNED_INT, (const void *)(2*cols*i*sizeof(unsigned int)));
	}
	glDisableVertexAttribArray(h_nor);
	glDisableVertexAttribArray(h_pos);
	glDisableVertexAttribArray(h_tex);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	MV->popMatrix();
}