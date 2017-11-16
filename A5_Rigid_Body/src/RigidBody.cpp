#define TETLIBRARY

#include <iostream>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "RigidBody.h"
#include "Particle.h"
#include "MatrixStack.h"
#include "Program.h"
#include "GLSL.h"

using namespace std;
using namespace Eigen;

#define STATE_SIZE  18
#define MODE 1
//MODE: 0 EULER 1 RK4

RigidBody::RigidBody(int id, int rot, int type, Vector3d pos, Vector3d v0, Vector3d omega0) {
	
	this->nbodies = 1;
	this->id = id;
	this->setPosition(pos);
	this->setVelocity(v0);
	this->setOmega(omega0);

	double r = 0.02;
	cr = 0.8;

	auto body = make_shared<RBState>();
	bodies.push_back(body);
	
	// Initial orientation
	if (rot == 1) {
		bodies[0]->R << 0.707, 0.707, 0,
						-0.707, 0.707, 0,
						0, 0, 1;
	}

	// Choose the type of the rigid body
	if (type == 1) {
		in.load_ply("cube");
	}
	if (type == 2) {
		in.load_ply("bunny33");
		// Compute the inertia using Meshlab
		bodies[0]->Ibody << 0.080223, 0.024555, 0.002083,
							0.024555, 0.103117, -0.000250,
							0.002083, -0.000250, 0.131980;
		bodies[0]->Ibodyinv = bodies[0]->Ibody.inverse();
		bodies[0]->L = bodies[0]->Ibody * bodies[0]->omega;
	}
	if (type == 3) {
		in.load_ply("icosahedron");
		bodies[0]->Ibody << 0.734, 0.0, 0.0,
							0.0, 0.734, 0.0,
							0.0, 0.0, 0.734;
		bodies[0]->Ibodyinv = bodies[0]->Ibody.inverse();
		bodies[0]->L = bodies[0]->Ibody * bodies[0]->omega;
	}
	if (type == 4) {
		in.load_ply("dodecahedron");
	}
	
	tetrahedralize("pqz", &in, &out);
	nVerts = out.numberofpoints;
	nTriFaces = out.numberoftrifaces;
	nEdges = out.numberofedges;
	
	for (int i = 0; i < nVerts; i++) {
		auto p = make_shared<Particle>();
		nodes.push_back(p);
		p->r = r;
		p->x0 << out.pointlist[3 * i], out.pointlist[3 * i + 1], out.pointlist[3 * i + 2];
		p->x = p->x0;
		p->v0 = bodies[0]->v;
		
		p->v = p->v0;
		p->m = 0.1;
		p->i = i;
		p->fixed = false;
	}
	t0 = 0.0;
	t = t0;

	// Build normals of box
	normal.resize(6, 3);
	normal.row(0) = Vector3d(1.0, 0.0, 0.0);
	normal.row(1) = Vector3d(-1.0, 0.0, 0.0);
	normal.row(2) = Vector3d(0.0, 1.0, 0.0);
	normal.row(3) = Vector3d(0.0, -1.0, 0.0);
	normal.row(4) = Vector3d(0.0, 0.0, 1.0);
	normal.row(5) = Vector3d(0.0, 0.0, -1.0);

	// Build vertex buffers
	posBuf.clear();
	norBuf.clear();
	texBuf.clear();
	eleBuf.clear();

	posBuf.resize(nTriFaces * 9);
	norBuf.resize(nTriFaces * 9);
	eleBuf.resize(nTriFaces * 3);

	updatePosNor();	
	for (int i = 0; i <  nTriFaces; i++) {
		eleBuf[3 * i + 0] = 3 * i + 0;
		eleBuf[3 * i + 1] = 3 * i + 1;
		eleBuf[3 * i + 2] = 3 * i + 2;
	}
}

void RigidBody::setVelocity(Eigen::Vector3d v) {
	bodies[0]->v = v;
	bodies[0]->P = bodies[0]->mass * v;
}

void RigidBody::setOmega(Eigen::Vector3d omega) {
	bodies[0]->omega = omega;
	bodies[0]->L = bodies[0]->Ibody * omega;
}

void RigidBody::setPosition(Eigen::Vector3d pos) {
	bodies[0]->x = pos;
}

// This test function is adapted from Moller-Trumbore intersection algorithm. 
// See https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm 

bool RigidBody::rayTriangleIntersects(Vector3d v1, Vector3d v2, Vector3d v3, Vector3d dir, Vector3d pos) {

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

void RigidBody::updatePosNor() {
	Matrix3d R = bodies[0]->R;
	Vector3d x = bodies[0]->x;

	for (int i = 0; i < nodes.size(); i++) {
		if (nodes[i]->i != -1) {
			nodes[i]->x_old = nodes[i]->x;
			nodes[i]->x = x + R * nodes[i]->x0;
		}
	}

	for (int iface = 0; iface < nTriFaces; iface++) {
		Vector3d p1 = nodes[out.trifacelist[3 * iface + 0]]->x;
		Vector3d p2 = nodes[out.trifacelist[3 * iface + 1]]->x;
		Vector3d p3 = nodes[out.trifacelist[3 * iface + 2]]->x;

		//Position
		Vector3d e1 = p2 - p1;
		Vector3d e2 = p3 - p1;
		Vector3d normal = e1.cross(e2);
		normal.normalize();
		
		for (int idx = 0; idx < 3; idx++) {
			posBuf[9 * iface + 0 + idx] = p1(idx);
			posBuf[9 * iface + 3 + idx] = p2(idx);
			posBuf[9 * iface + 6 + idx] = p3(idx);
			norBuf[9 * iface + 0 + idx] = normal(idx);
			norBuf[9 * iface + 3 + idx] = normal(idx);
			norBuf[9 * iface + 6 + idx] = normal(idx);
		}
	}

	// Update AABB Bounding Box
	computeAABB();
}

void RigidBody::normalizeR() {
	for (int i = 0; i < nbodies; i++) {
		Vector3d x = bodies[i]->R.col(0);
		Vector3d y = bodies[i]->R.col(1);
		Vector3d z = bodies[i]->R.col(2);

		x.normalize();
		y = z.cross(x);
		y.normalize();
		z = x.cross(y);
		z.normalize();

		bodies[i]->R.col(0) = x;
		bodies[i]->R.col(1) = y;
		bodies[i]->R.col(2) = z;
	}
}

Eigen::Matrix3d RigidBody::Star(Eigen::Vector3d a) {
	Matrix3d s;
	s << 0, -a(2), a(1),
		a(2), 0, -a(0),
		-a(1), a(0), 0;
	return s;
}

void RigidBody::ddtstate2array(std::shared_ptr<RBState> rb, Eigen::VectorXd &xdot) {
	// copy dxdt = vt into xdot
	xdot.segment<3>(0) = rb->v;

	// compute R*=w*R
	Eigen::Matrix3d Rdot = Star(rb->omega) * rb->R;

	// copy R* into vector
	for (int i = 0; i < 3; i++) {
		xdot.segment<3>(3 + 3 * i) = Rdot.row(i);
	}

	// dpdt = F
	xdot.segment<3>(12) = rb->force;
	xdot.segment<3>(15) = rb->torque;
}

void RigidBody::Dxdt(double t, Eigen::VectorXd x, Eigen::VectorXd &xdot) {
	// put data in x into bodies
	array2bodies(x);
	for (int i = 0; i < nbodies; i++) {
		VectorXd s(STATE_SIZE);
		computeForceTorque(t, bodies[i]);
		ddtstate2array(bodies[i], s);
		xdot.segment<STATE_SIZE>(i*STATE_SIZE) = s;
	}
}

// Copy information from the state variables to an array
void RigidBody::state2array(std::shared_ptr<RBState> rb, Eigen::VectorXd &y) {
	y.resize(18);
	y.segment<3>(0) = rb->x; // component of position
	// copy rotation matrix
	for (int i = 0; i < 3; i++) {
		y.segment<3>(3 + 3 * i) = rb->R.row(i);
	}
	y.segment<3>(12) = rb->P;
	y.segment<3>(15) = rb->L;
}

// Copy information from an array into the state variables
void RigidBody::array2state(std::shared_ptr<RBState> rb, Eigen::VectorXd y) {

	rb->x = y.segment<3>(0);
	for (int i = 0; i < 3; i++) {
		rb->R.row(i) = y.segment<3>(3 + 3 * i);
	}
	rb->P = y.segment<3>(12);
	rb->L = y.segment<3>(15);
	
	// compute auxiliary variables
	rb->v = rb->P / rb->mass;
	rb->Iinv = rb->R * rb->Ibodyinv * rb->R.transpose();
	rb->omega = rb->Iinv * rb->L;
}

void RigidBody::array2bodies(Eigen::VectorXd x) {
	for (int i = 0; i < nbodies; i++) {
		array2state(bodies[i], x.segment<STATE_SIZE>(STATE_SIZE * i));
	}
}

void RigidBody::bodies2array(Eigen::VectorXd &x) {
	x.resize(STATE_SIZE * nbodies);
	for (int i = 0; i < nbodies; i++) {
		VectorXd s;
		state2array(bodies[i], s);
		x.segment<STATE_SIZE>(STATE_SIZE * i) = s;
	}
}

void RigidBody::computeForceTorque(double t, std::shared_ptr<RBState> rb) {
	rb->force = Vector3d(0.0, -1.0, 0.0);
	rb->torque = Vector3d(0.0, 0.0, 0.0);
}

// Compute the AABB bounding box and save the index of vertices
void RigidBody::computeAABB() {
	max3 = Vector3d(INT_MIN + 0.01, INT_MIN + 0.01, INT_MIN + 0.01);
	min3 = Vector3d(INT_MAX - 0.01, INT_MAX -0.01, INT_MAX - 0.01);
	index_max3.setZero();
	index_min3.setZero();

	for (int axis = 0; axis < 3; axis++) {
		for (int i = 0; i < nodes.size(); i++) {

			if (nodes[i]->x(axis) > max3(axis)) {
				max3(axis) = nodes[i]->x(axis);
				// save the node index for collisions
				index_max3(axis) = i;
			}

			if (nodes[i]->x(axis) < min3(axis)) {
				min3(axis) = nodes[i]->x(axis);
				// save the node index for collisions
				index_min3(axis) = i;
			}
		}
	}	
}

// Check collsions with box
void RigidBody::collideBox(VectorXd box) {
	double obj_m, vminus, j;
	Vector3d ra, v_pa, J;
	int id_pa;
	Vector3d nor;
	
	// Detect collsions with lower bounds
	for (int i = 0; i < 3; i++) {
		obj_m = min3(i);
		id_pa = index_min3(i);
		ra = nodes[id_pa]->x - bodies[0]->x;
		v_pa = bodies[0]->v + bodies[0]->omega.cross(ra);
		nor = normal.row(2 * i);
		vminus = v_pa.dot(nor);

		if (obj_m < box(2 * i) && vminus < -0.001) {
			j = (-(1.0 + cr) * vminus) / (1.0 / bodies[0]->mass + nor.dot( bodies[0]->Iinv * (ra.cross(nor)).cross(ra) ));

			// Compute impulse
			J = j * nor;
			bodies[0]->P += J;
			bodies[0]->L += ra.cross(J);

			bodies[0]->v = bodies[0]->P / bodies[0]->mass;
			bodies[0]->omega = bodies[0]->Iinv * bodies[0]->L;	
		}
	}
	// Detect collsions with upper bounds
	for (int i = 0; i < 3; i++) {
		obj_m = max3(i);
		id_pa = index_max3(i);
		ra = nodes[id_pa]->x - bodies[0]->x;
		v_pa = bodies[0]->v + bodies[0]->omega.cross(ra);
		nor = normal.row(2*i+1);
		vminus = v_pa.dot(nor);

		if (obj_m > box(2*i+1) && vminus < -0.001) {
			j = (-(1.0 + cr) * vminus) / (1.0 / bodies[0]->mass + nor.dot(bodies[0]->Iinv * (ra.cross(nor)).cross(ra)));

			// Compute impulse
			J = j * nor;
			bodies[0]->P += J;
			bodies[0]->L += ra.cross(J);

			bodies[0]->v = bodies[0]->P / bodies[0]->mass;
			bodies[0]->omega = bodies[0]->Iinv * bodies[0]->L;
		}
	}
}

void RigidBody::collideRBs(const vector< shared_ptr<RigidBody> >rigidbodies, int checkedge) {

	for (int i = 0; i < rigidbodies.size(); i++) {

		shared_ptr<RigidBody> rb = rigidbodies[i];
		int num = 0;

		// Check if the checked one is the same rigid body
		if (rb->id != id) {

			// Check AABB Bounding Box
			for (int axis = 0; axis < 3; axis ++) {
				if (rb->min3(axis) > max3(axis) || rb->max3(axis) < min3(axis)) {
					// No collsion in this axis
				}
				else {
					// Might collide 
					num = num + 1;
				}
			}

			// If the two rbs collide in all three axies, they might collide with each other
			if (num == 3) {
				
				// Check Vertex to Face collsions
				if (checkedge != 2) {
					for (int iface = 0; iface < nTriFaces; iface++) {
						// face vertices
						Vector3d p1 = nodes[out.trifacelist[3 * iface + 0]]->x;
						Vector3d p2 = nodes[out.trifacelist[3 * iface + 1]]->x;
						Vector3d p3 = nodes[out.trifacelist[3 * iface + 2]]->x;

						Vector3d nor = -(p1 - p3).cross(p2 - p3);
						nor.normalize();

						for (int inode = 0; inode < rb->nodes.size(); inode++) {
							shared_ptr<Particle> p = rb->nodes[inode];
							Vector3d x_old = p->x_old;
							Vector3d x_new = p->x;
							Vector3d dir = x_new - x_old;
							double f = (x_old - p1).dot(nor) / ((x_old - p1).dot(nor) - (x_new - p1).dot(nor));

							if (rayTriangleIntersects(p1, p2, p3, dir, x_old) && f < 1.0) {
								// Collide with the face

								Vector3d r_a = x_new - rb->bodies[0]->x;
								Vector3d va = rb->bodies[0]->v + rb->bodies[0]->omega.cross(r_a);
								Vector3d r_b = x_new - bodies[0]->x;
								Vector3d vb = bodies[0]->v + bodies[0]->omega.cross(r_b);
								double vminus_rel = nor.dot(va - vb);
								double j = (-(1.0 + cr) * vminus_rel) /
									(1.0 / bodies[0]->mass + 1.0 / rb->bodies[0]->mass
										+ nor.dot(rb->bodies[0]->Iinv * (r_a.cross(nor)).cross(r_a) + bodies[0]->Iinv * (r_b.cross(nor)).cross(r_b))
										);

								// Compute impulse
								Vector3d J = j * nor;

								// For one obj b
								bodies[0]->P -= J;
								bodies[0]->L -= r_b.cross(J);

								bodies[0]->v = bodies[0]->P / bodies[0]->mass;
								bodies[0]->omega = bodies[0]->Iinv * bodies[0]->L;

								// For the other a
								rb->bodies[0]->P += J;
								rb->bodies[0]->L += r_a.cross(J);

								rb->bodies[0]->v = rb->bodies[0]->P / rb->bodies[0]->mass;
								rb->bodies[0]->omega = rb->bodies[0]->Iinv * rb->bodies[0]->L;
							}
						}
					}
				}// End of Vertex to Face collsion check

				// Check Edge to Edge collsions
				if (checkedge == 1) {
					for (int iedge = 0; iedge < nEdges; iedge++) {
						Vector3d v1, v2, edgei, va, vb, edgej, nor1, r, pa, qa, m;
						double s, t;

						v1 = nodes[out.edgelist[2 * iedge + 0]]->x;
						v2 = nodes[out.edgelist[2 * iedge + 1]]->x;
						edgei = v1 - v2;

						for (int jedge = 0; jedge < rb->nEdges; jedge++) {
							va = rb->nodes[rb->out.edgelist[2 * jedge + 0]]->x;
							vb = rb->nodes[rb->out.edgelist[2 * jedge + 1]]->x;
							edgej = va - vb;

							//nor1 is the direction of paqa
							nor1 = edgej.cross(edgei).normalized();

							// vector from p1 to q1
							r = v2 - va;

							// the value of s corresponding to pa is
							s = r.dot(edgei.normalized().cross(nor1)) / edgej.dot(edgei.normalized().cross(nor1));
							// the value of t corresponding to qa is
							t = -r.dot(edgej.normalized().cross(nor1)) / edgei.dot(edgej.normalized().cross(nor1));

							// compute the position of collision
							pa = va + s * edgej;
							qa = v2 + t * edgei;
							m = qa - pa;

							if (m.norm() < 0.01 && (s > 0 && s < 1 || s == 0 || s == 1) && (t > 0 && t < 1 || t == 0 || t == 1)) {

								Vector3d r_a = qa - bodies[0]->x;
								Vector3d r_b = pa - rb->bodies[0]->x;
								Vector3d va = bodies[0]->v + bodies[0]->omega.cross(r_a);

								Vector3d vb = rb->bodies[0]->v + rb->bodies[0]->omega.cross(r_b);

								double vminus_rel = nor1.dot(va - vb);
								double j = (-(1.0 + cr) * vminus_rel) /
									(1.0 / bodies[0]->mass + 1.0 / rb->bodies[0]->mass
										+ nor1.dot(rb->bodies[0]->Iinv * (r_b.cross(nor1)).cross(r_b) + bodies[0]->Iinv * (r_a.cross(nor1)).cross(r_a))
										);

								// Compute impulse
								Vector3d J = j * nor1;

								// For one obj a
								//bodies[0]->P -= J;
								//bodies[0]->L += r_a.cross(J);

								//bodies[0]->v = bodies[0]->P / bodies[0]->mass;
								//bodies[0]->omega = bodies[0]->Iinv * bodies[0]->L;

								//// For the other obj b
								//rb->bodies[0]->P += J;
								//rb->bodies[0]->L -= r_b.cross(J);

								//rb->bodies[0]->v = rb->bodies[0]->P / rb->bodies[0]->mass;
								//rb->bodies[0]->omega = rb->bodies[0]->Iinv * rb->bodies[0]->L;

								// For one obj a
								bodies[0]->P += J;
								bodies[0]->L += r_a.cross(J);

								bodies[0]->v = bodies[0]->P / bodies[0]->mass;
								bodies[0]->omega = bodies[0]->Iinv * bodies[0]->L;

								// For the other obj b
								rb->bodies[0]->P -= J;
								rb->bodies[0]->L -= r_b.cross(J);

								rb->bodies[0]->v = rb->bodies[0]->P / rb->bodies[0]->mass;
								rb->bodies[0]->omega = rb->bodies[0]->Iinv * rb->bodies[0]->L;
							}
						}
					}
				}
			}
		}
	}
}

void RigidBody::step(double h, const Eigen::Vector3d &grav, const vector< shared_ptr<RigidBody> > rigidbodies, int checkedge) {

	VectorXd box(6);
	box << -10.0, 10.0, -10.0, 10.0, -10.0, 10.0;

	t = t + h;

	// Euler
	if (MODE == 0) {
		RBState newstate;
		VectorXd xold;
		bodies2array(xold);
	
		VectorXd xdot;
		xdot = xold;
		Dxdt(t, xold, xdot);

		VectorXd xnew;
		xnew = xold + xdot * h;
		array2bodies(xnew);
	}

	// RK4
	if (MODE == 1) {
		VectorXd xold;
		bodies2array(xold);

		VectorXd xdot;
		xdot = xold;
		Dxdt(t, xold, xdot);

		VectorXd K1;
		K1 = xdot * h;

		Dxdt(t, xold + 1.0 / 2.0 * K1, xdot);
		VectorXd K2 = xdot * h;

		Dxdt(t, xold + 1.0 / 2.0 * K2, xdot);
		VectorXd K3 = xdot * h;

		Dxdt(t, xold + K3, xdot);
		VectorXd K4 = xdot * h;

		VectorXd xnew;
		xnew = xold + 1.0 / 6.0 * (K1 + 2 * K2 + 2 * K3 + K4);
		array2bodies(xnew);
	}

	// Collision with box
	collideBox(box);

	// Collision with other rigidbodies
	collideRBs(rigidbodies, checkedge);

	updatePosNor();

	normalizeR();
}

double RigidBody::randDouble(double l, double h)
{
	
	float r = rand() / (double)RAND_MAX;
	return (1.0 - r) * l + r * h;
}

void RigidBody::init() {
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &norBufID);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &eleBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, eleBuf.size() * sizeof(unsigned int), &eleBuf[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	assert(glGetError() == GL_NO_ERROR);
}

void RigidBody::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> p)const {

	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(1.0, 0.0, 0.0).data());
	glUniform3fv(p->getUniform("kdBack"), 1, Vector3f(1.0, 1.0, 0.0).data());
	MV->pushMatrix();
	glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
	int h_pos = p->getAttribute("aPos");
	glEnableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);

	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);

	int h_nor = p->getAttribute("aNor");
	glEnableVertexAttribArray(h_nor);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_nor, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glDrawElements(GL_TRIANGLES, 3 * nTriFaces, GL_UNSIGNED_INT, (const void *)(0 * sizeof(unsigned int)));

	// Draw the AABB bounding box
	drawAABB();

	glDisableVertexAttribArray(h_nor);
	glDisableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	MV->popMatrix();
}

void RigidBody::drawAABB() const {
	glDisable(GL_LIGHTING);
	glLineWidth(1);
	glColor3f(0.5f, 0.5f, 0.5f);
	glBegin(GL_LINE_LOOP);
	glNormal3f(0, -1.0f, 0);
	glVertex3f(min3[0], min3[1], min3[2]); //0
	glVertex3f(max3[0], min3[1], min3[2]); //3
	glVertex3f(max3[0], min3[1], max3[2]); //2
	glVertex3f(min3[0], min3[1], max3[2]); //1
	glVertex3f(min3[0], max3[1], max3[2]); //5
	glVertex3f(max3[0], max3[1], max3[2]); //6
	glVertex3f(max3[0], max3[1], min3[2]); //7
	glVertex3f(min3[0], max3[1], min3[2]); //4
	glEnd();

	glBegin(GL_LINES);
	glNormal3f(0, -1.0f, 0);
	glVertex3f(min3[0], max3[1], min3[2]); //4
	glVertex3f(min3[0], max3[1], max3[2]); //5
	glVertex3f(min3[0], min3[1], min3[2]); //0
	glVertex3f(min3[0], min3[1], max3[2]); //1
	glVertex3f(max3[0], max3[1], max3[2]); //6
	glVertex3f(max3[0], min3[1], max3[2]); //2
	glVertex3f(max3[0], max3[1], min3[2]); //7
	glVertex3f(max3[0], min3[1], min3[2]); //3
	glEnd();
	glLineWidth(1);
	glEnable(GL_LIGHTING);
}

RigidBody::~RigidBody() {
}