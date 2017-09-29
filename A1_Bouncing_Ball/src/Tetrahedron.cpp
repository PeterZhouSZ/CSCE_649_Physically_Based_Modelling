#include <iostream>

#include "Tetrahedron.h"
#include "Particle.h"
#include "Program.h"
#include "GLSL.h"
#include "MatrixStack.h"
#include "Spring.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
using namespace Eigen;

Tetrahedron::Tetrahedron(
	const Eigen::Vector3d &x0,
	const Eigen::Vector3d &x1,
	const Eigen::Vector3d &x2,
	const Eigen::Vector3d &x3,
	double mass,
	double stiffness,
	const Eigen::Vector2d &damping)
{
	assert(mass > 0.0);
	assert(stiffness > 0.0);
	
	this->damping = damping;
	this->young = 0.01;
	this->poisson = 0.3;

	double r = 0.02; // Used for collisions
	int nVerts = 4;

	MatrixXd X(4, 3);
	X.row(0) = x0;
	X.row(1) = x1;
	X.row(2) = x2;
	X.row(3) = x3;
		 
	double mass_n = mass / (nVerts);
	n = 0;
	// Create particles
	for (int i = 0; i < nVerts; i++) {
		auto p = make_shared<Particle>();
		particles.push_back(p);
		p->r = r;
		p->x = X.row(i); // Initialize pos of nodes
		p->v << 0.0, 0.0, 0.0;
		p->m = mass_n;
		p->i = i;
		p->fixed = false;
		n+=3;
	}
	
	// Build system matrices and vectors
	M.resize(n, n);
	K.resize(n, n);
	v.resize(n);
	f.resize(n);

	// Precompute X_inv = [x1,x2,x3]-1, constant throughout the simulation
	MatrixXd _X_inv(3, 3);
	Vector3d xb = x1 - x0;
	Vector3d xc = x2 - x0;
	Vector3d xd = x3 - x0;
	_X_inv.col(0) = xb;
	_X_inv.col(1) = xc;
	_X_inv.col(2) = xd;
	X_inv = _X_inv.inverse(); 

	// 
	// Build vertex buffers
	posBuf.clear();
	norBuf.clear();
	texBuf.clear();
	eleBuf.clear();
	posBuf.resize(nVerts * 3);
	norBuf.resize(nVerts * 3);
	updatePosNor();

	//// Texture coordinated
	
	texBuf = { 0, 0, 0, 1, 1, 1, 1, 0 };

	// Elements
	eleBuf =  { 
		0, 2, 1,
		1, 2, 3,
		2, 0, 3,
		3, 0, 2};

}

MatrixXd Tetrahedron::hooke(double young, double poisson) {
	double v = poisson;
	double v1 = 1 - v;
	double v2 = 1 - 2 * v;
	double s = young / ((1 + v)*(1 - 2 * v));

	MatrixXd E(6, 6);
	E <<
		v1, v, v, 0, 0, 0,
		v, v1, v, 0, 0, 0,
		v, v, v1, 0, 0, 0,
		0, 0, 0, v2, 0, 0,
		0, 0, 0, 0, v2, 0,
		0, 0, 0, 0, 0, v2;

	E = s*E;
	return E;
}


void Tetrahedron::step(double h, const Vector3d &grav) {
	M.setZero();
	K.setZero();
	v.setZero();
	f.setZero();

	for (int i = 0; i < (int)particles.size(); i++) {
		int idx = particles[i]->i;
		double mass = particles[i]->m;
		
		Matrix3d A;
		A << mass, 0, 0,
			0, mass, 0,
			0, 0, mass;
		M.block<3, 3>(3*idx, 3*idx) = A; // filling M

		Vector3d B;
		B << particles[i]->v;
		v.segment<3>(3*idx) = B; // filling v 
		f.segment<3>(3*idx) = mass * grav; // filling f with fg
	}

	MatrixXd dp(3, 3);
	Vector3d pb = particles[1]->x - particles[0]->x;
	Vector3d pc = particles[2]->x - particles[0]->x;
	Vector3d pd = particles[3]->x - particles[0]->x;
	dp.col(0) = pb;
	dp.col(1) = pc;
	dp.col(2) = pd;
	// Compute Deformation Gradient
	MatrixXd P = dp * X_inv;

	Matrix3d I;
	I.setIdentity();
	MatrixXd du = P - I;

	cout << du << endl;
	// Compute the strain
	Matrix3d strain_m = 0.5 * (du + du.transpose() + du.transpose() * du);

	cout << strain_m << endl;

	VectorXd strain_v(6);
	strain_v << strain_m(0, 0), strain_m(1, 1), strain_m(2, 2), 
		strain_m(0, 1), strain_m(1, 2), strain_m(0, 2);
	
	MatrixXd E = hooke(young, poisson);

	//cout << E << endl;

	// Compute the stress
	VectorXd stress_v = E * strain_v;
	cout << stress_v << endl;


	Matrix3d stress_m;
	stress_m <<
		stress_v(0), stress_v(3), stress_v(5),
		stress_v(3), stress_v(1), stress_v(4),
		stress_v(4), stress_v(5), stress_v(2);

	MatrixXi triIndices(4, 3);
	triIndices<< 
		0, 2, 1,
		1, 2, 3,
		2, 0, 3,
		3, 0, 2;

	for (int i = 0; i < 4; i++) {
		
		int ia = triIndices(i, 0);
		int ib = triIndices(i, 1);
		int ic = triIndices(i, 2);

		Vector3d pa = particles[ia]->x;
		Vector3d pb = particles[ib]->x;
		Vector3d pc = particles[ic]->x;

		Vector3d fk = -stress_m * ((pb - pa).cross(pc - pa));
		cout << fk << endl << endl;
		// filling f with fk
		for (int i = 0; i < 3; i++) {
			f(3 * ia + i) += fk(i) / 3;
			f(3 * ib + i) += fk(i) / 3;
			f(3 * ic + i) += fk(i) / 3;
		}

	}
	double damp = 0.9;
	cout << f << endl << endl;

	VectorXd v_new = (M).ldlt().solve(M*v + h*f);

	// Update velocity
	for (int i = 0; i < (int)particles.size(); i++) {
		if (particles[i]->i != -1) {
			particles[i]->v = v_new.segment<3>(particles[i]->i);
		}
	}

	// Update position
	for (int i = 0; i < (int)particles.size(); i++) {
		if (particles[i]->i != -1) {
			particles[i]->x += particles[i]->v *h;
		}
		cout << particles[i]->x << endl;
	}

	cout << "v_new" << v_new << endl << endl;

	updatePosNor();

}


void Tetrahedron::init() {
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);

	//glGenBuffers(1, &posBufID);
	//glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	//glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &texBufID);
	glBindBuffer(GL_ARRAY_BUFFER, texBufID);
	glBufferData(GL_ARRAY_BUFFER, texBuf.size() * sizeof(float), &texBuf[0], GL_STATIC_DRAW);

	glGenBuffers(1, &eleBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, eleBuf.size() * sizeof(unsigned int), &eleBuf[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	assert(glGetError() == GL_NO_ERROR);

}


void Tetrahedron::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> p) const
{
	
	// Draw mesh
	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(1.0, 0.0, 0.0).data());
	glUniform3fv(p->getUniform("kdBack"), 1, Vector3f(1.0, 1.0, 0.0).data());
	MV->pushMatrix();

	glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	
	int h_pos = p->getAttribute("aPos");
	glEnableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);

	glDrawElements(GL_TRIANGLES, 12, GL_UNSIGNED_INT, (const void *)(0  * sizeof(unsigned int)));
	
	glDisableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	MV->popMatrix();
}

void Tetrahedron::tare() {
	for (int k = 0; k < (int)particles.size(); k++) {
		particles[k]->tare();
	}
}

void Tetrahedron::reset() {
	for (int k = 0; k < (int)particles.size(); k++) {
		particles[k]->reset();
	}
	updatePosNor();
}

void Tetrahedron::updatePosNor()
{
	// Position
	for (int i = 0; i < (int)particles.size(); i++) {
		Vector3d x = particles[i]->x;
		posBuf[3 * i + 0] = x(0);
		posBuf[3 * i + 1] = x(1);
		posBuf[3 * i + 2] = x(2);
	}

	// Normal
	// first triangles
	//Vector3d dx0 = particles[2]->x - particles[0]->x;
	//Vector3d dx1 = particles[1]->x - particles[0]->x;
	//Vector3d c = dx0.cross(dx1);



}

Tetrahedron::~Tetrahedron()
{
}
