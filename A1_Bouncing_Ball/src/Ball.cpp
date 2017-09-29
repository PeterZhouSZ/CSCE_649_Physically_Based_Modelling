#include <iostream>
#include <cmath>
#include <vector>
#include <math.h>
#include <ctime> // Needed for the true randomization
#include <cstdlib> 

#include "Box.h"
#include "Ball.h"
#include "Particle.h"
#include "Program.h"
#include "GLSL.h"
#include "MatrixStack.h"
#include "Spring.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
using namespace Eigen;

#define M_PI       3.14159265358979323846
#define M_PI_2     1.57079632679489661923
#define EPSILON    0.000001


Ball::Ball(
	const Eigen::Vector3d &center,
	const Eigen::Vector3d &velocity,
	double m,
	double radius) {

	assert(m > 0.0);
	assert(radius > 0.0);

	this->r = radius;
	this->m = m;
	this->center = center;
	this->v = velocity;
	this->rings = 20;
	this->sectors = 20;
	this->center_0 = center;
	this->v0 = velocity;

	// Build vertex buffers
	posBuf.clear();
	norBuf.clear();
	//texBuf.clear();
	eleBuf.clear();
	posBuf.resize(rings*sectors*3);
	norBuf.resize(rings*sectors*3);
	//texBuf.resize(rings *sectors* 2);
	eleBuf.resize(rings*sectors * 4);
	updatePosNor();
}

void Ball::updatePosNor() {
	int ri, s;
	float const R = 1. / (float)(rings - 1);
	float const S = 1. / (float)(sectors - 1);

	// Position and Normal
	for (ri = 0; ri < rings; ri++) {
		for (s = 0; s < sectors; s++) {
			int k = ri * sectors + s;
			float const y = sin(-M_PI_2 + M_PI * ri * R);
			float const x = cos(2 * M_PI * s *S)* sin(M_PI * ri *R);
			float const z = sin(2 * M_PI * s * S)* sin(M_PI*ri*R);
			 
			//texBuf[2*k+0] = s*S;
			//texBuf[2*k+1] = ri*R;

			posBuf[3*k+0] = x*r+center[0];
			posBuf[3*k+1] = y*r+center[1];
			posBuf[3*k+2] = z*r+center[2];

			norBuf[3*k+0]= x;
			norBuf[3*k+1]= y;
			norBuf[3*k+2]= z;

		}
	}

	// Update Element Index Buffer
	for (ri = 0; ri < rings - 1; ri++) {
		for (s = 0; s < sectors - 1; s++) {
			int k = ri*(sectors - 1) + s;
			eleBuf[4*k+0]=ri*sectors+s;
			eleBuf[4*k+1]=ri*sectors + (s + 1);
			eleBuf[4*k+2]=(ri + 1)*sectors + (s + 1);
			eleBuf[4*k+3]=(ri+1)*sectors+s;

		}
	}
}

void Ball::step_poly(double h, const Vector3d &grav, const std::shared_ptr<Box> box) {
	double coef_air = 1;
	double epsilon = -0.95;
	double mu = 0.1;
	double hremaining = h;
	double timestep = hremaining;

	Vector3d k(0, 0, 15);
	Vector3d m(15, 0, 0);
	Vector3d n(0, 15, 0);

	// Get the normal of the plane
	Vector3d normal_kmn = -(k - n).cross(m - n); 
	double twoA = normal_kmn.norm();
	Vector3d nor = normal_kmn / twoA;

	while (hremaining > 0.00001) {
		double fraction;
		Vector3d normal;

		// Euler Integration
		int num_cols = 0;
		Vector3d v_new = v + hremaining * (grav - coef_air * v);
		Vector3d center_new = center + hremaining * v_new; //
 // Collision Detection for six faces

		int col_r = ((center_new[0] - box->right + r) > EPSILON);
		int col_l = ((center_new[0] - box->left - r) < -EPSILON);
		int col_u = ((center_new[1] - box->up + r) > EPSILON);
		int col_d = ((center_new[1] - box->down - r) < -EPSILON);
		int col_f = ((center_new[2] - box->forward + r) > EPSILON);
		int col_b = ((center_new[2] - box->back - r) < -EPSILON);

		// care for r
		//double t_kmn = (k - center + nor*r).dot(normal_kmn) / (v.dot(normal_kmn));
		//int col_kmn = (t_kmn >-EPSILON) && (t_kmn - h < -EPSILON);
		int col_kmn = ((center_new - k).dot(nor) - r<-EPSILON);
		//int col_l = ((center_new[0] - r) < -EPSILON);
		//int col_d = ((center_new[1] - r) < -EPSILON);
		//int col_b = ((center_new[2] - r) < -EPSILON);
		double f_r = (center[0] - box->right + r) / ((center[0] - box->right) - (center_new[0] - box->right));
		double f_l = (center[0] - box->left - r) / ((center[0] - box->left) - (center_new[0] - box->left));
		double f_u = (center[1] - box->up + r) / ((center[1] - box->up) - (center_new[1] - box->up));
		double f_d = (center[1] - box->down - r) / ((center[1] - box->down) - (center_new[1] - box->down));
		double f_f = (center[2] - box->forward + r) / ((center[2] - box->forward) - (center_new[2] - box->forward));
		double f_b = (center[2] - box->back - r) / ((center[2] - box->back) - (center_new[2] - box->back));

		num_cols = col_r + col_l + col_u + col_d + col_f + col_b + col_kmn;

		if (num_cols != 0) {
			// Determination 
			//Vector3d kmn_hit = center + nor * r + t_kmn * v;
			double f_kmn = ((center - k).dot(nor) - r) / ((center - k).dot(nor) - (center_new - k).dot(nor));
			//double f_l = (center[0] - r) / (center[0] - center_new[0]);
			//double f_d = (center[1] - r) / (center[1] - center_new[1]);
			//double f_b = (center[2] - r) / (center[2] - center_new[2]);

			// To check point kmn_hit inside or outside the triangle using barycentric coordinated
			if (col_kmn) {
				Vector3d vc = v + f_kmn * h * (grav - coef_air * v);
				Vector3d center_c = center + f_kmn* h * v;
				Vector3d kmn_hit = center_c + r * nor;
				double uu = ((k - m).cross(kmn_hit - m)).dot(nor) / twoA;
				double vv = ((n - k).cross(kmn_hit - k)).dot(nor) / twoA;
				double ww = 1 - uu - vv;
				if (uu > -EPSILON && vv > -EPSILON && ww > -EPSILON) {

					// Response

					//Vector3d vc = v + fraction * h * (grav - coef_air * v);
					//Vector3d center_c = center + fraction * h * v;

					//Vector3d vc_n = (v.dot(nor))*nor;
					//Vector3d vc_t = v - vc_n;
					//Vector3d v_new_n = epsilon * vc_n;
					//Vector3d v_new_t = (1 - mu)* vc_t;

					Vector3d vc_n = (vc.dot(nor))*nor;
					Vector3d vc_t = vc - vc_n;
					Vector3d v_new_n = epsilon * vc_n;
					Vector3d v_new_t = (1 - mu)* vc_t;

					v_new = v_new_n + v_new_t;
					center_new = center_c;
					v = v_new;
					center = center_new;
					hremaining = hremaining - timestep;
					updatePosNor();

					//timestep = t_kmn;
					timestep = f_kmn * timestep;
				}
			}
			else {

				//uu > -EPSILON && vv > -EPSILON && ww > -EPSILON
				// Inside the face kmn, and collision happened there
				if (col_r) {
					normal << -1, 0, 0;
					fraction = f_r;
				}
				if (col_l) {
					normal << 1, 0, 0;
					fraction = f_l;
				}
				if (col_u) {
					normal << 0, -1, 0;
					fraction = f_u;
				}
				if (col_d) {
					normal << 0, 1, 0;
					fraction = f_d;
				}
				if (col_f) {
					normal << 0, 0, -1;
					fraction = f_f;
				}
				if (col_b) {
					normal << 0, 0, 1;
					fraction = f_b;
				}

				// Response
				timestep = fraction * timestep;
				Vector3d vc = v + fraction * h * (grav - coef_air * v);
				Vector3d center_c = center + fraction * h * v;

				Vector3d vc_n = (vc.dot(normal))*normal;
				Vector3d vc_t = vc - vc_n;
				Vector3d v_new_n = epsilon * vc_n;
				Vector3d v_new_t = (1 - mu)* vc_t;

				v_new = v_new_n + v_new_t;
				center_new = center_c;
				v = v_new;
				center = center_new;
				hremaining = hremaining - timestep;
				updatePosNor();

			}
		}else {
			// If there is no collision 
			hremaining = hremaining - timestep;
			v = v_new;	
			center = center_new;
			updatePosNor();
		}
	}
}


void Ball::step(double h, const Vector3d &grav, const std::shared_ptr<Box> box) {
	// Test Rest..
	/*if (abs(v[1]) <= 0.01&& center[1] - r< 0.1) {
		v[1] = 0;
		center[1] = r;
		updatePosNor();

	}
	else {*/
		
		double coef_air = 1;    // Air friction coefficient
		double epsilon = -0.95;   // Elascity  
		double mu = 0.1;		// Friction
		double hremaining = h;
		double timestep = hremaining;

		while (hremaining > 0.00001) {

			double fraction;
			Vector3d normal;
			// Euler Integration
			int num_cols = 0;
			Vector3d v_new = v + hremaining * (grav - coef_air * v);
			Vector3d center_new = center + hremaining * v_new; //

			// Collision Detection for six faces

			int col_r = ((center_new[0] - box->right + r) > EPSILON);
			int col_l = ((center_new[0] - box->left - r) < -EPSILON);
			int col_u = ((center_new[1] - box->up + r) > EPSILON);
			int col_d = ((center_new[1] - box->down - r) < -EPSILON);
			int col_f = ((center_new[2] - box->forward + r) > EPSILON);
			int col_b = ((center_new[2] - box->back - r) < -EPSILON);

			/* Collision Detection: sign(dn)?=sign(dn+1) if it is opposite sign, then are at opposite plane

			int col_r = ((center_new[0] - box->right + r)* (center[0] - box->right + r) < -EPSILON);
			int col_l = ((center_new[0] - box->left - r)*(center[0] - box->left - r) < -EPSILON);
			int col_u = ((center_new[1] - box->up + r)*(center[1] - box->up + r) < -EPSILON);
			int col_d = ((center_new[1] - box->down - r)*(center[1] - box->down - r) < -EPSILON);
			int col_f = ((center_new[2] - box->forward + r)*(center[2] - box->forward + r) < -EPSILON);
			int col_b = ((center_new[2] - box->back - r)*(center[2] - box->back - r) < -EPSILON);
			*/

			// Determination 
			double f_r = (center[0] - box->right + r) / ((center[0] - box->right) - (center_new[0] - box->right));
			double f_l = (center[0] - box->left - r) / ((center[0] - box->left) - (center_new[0] - box->left));
			double f_u = (center[1] - box->up + r) / ((center[1] - box->up) - (center_new[1] - box->up));
			double f_d = (center[1] - box->down - r) / ((center[1] - box->down) - (center_new[1] - box->down));
			double f_f = (center[2] - box->forward + r) / ((center[2] - box->forward) - (center_new[2] - box->forward));
			double f_b = (center[2] - box->back - r) / ((center[2] - box->back) - (center_new[2] - box->back));

			num_cols = col_r + col_l + col_u + col_d + col_f + col_b;

			if (num_cols != 0) {

				if (col_r) {
					normal << -1, 0, 0;
					fraction = f_r;
				}
				if (col_l) {
					normal << 1, 0, 0;
					fraction = f_l;
				}
				if (col_u) {
					normal << 0, -1, 0;
					fraction = f_u;
				}
				if (col_d) {
					normal << 0, 1, 0;
					fraction = f_d;
				}
				if (col_f) {
					normal << 0, 0, -1;
					fraction = f_f;
				}
				if (col_b) {
					normal << 0, 0, 1;
					fraction = f_b;
				}

				// Response
				timestep = fraction * timestep;
				Vector3d vc = v + fraction * h * (grav - coef_air * v);
				Vector3d center_c = center + fraction * h * v;

				Vector3d vc_n = (vc.dot(normal))*normal;
				Vector3d vc_t = vc - vc_n;
				Vector3d v_new_n = epsilon * vc_n;
				Vector3d v_new_t = (1 - mu)* vc_t;

				v_new = v_new_n + v_new_t;
				center_new = center_c;
				v = v_new;
				center = center_new;
				hremaining = hremaining - timestep;
				updatePosNor();

			}
			else {
				// If there is no collision 
				hremaining = hremaining - timestep;
				v = v_new;
				//cout << v << endl << endl;
				center = center_new;
				//cout << center << endl << endl;
				updatePosNor();
			}
		}

		//updatePosNor();
	}

	// More complex cases:
	// When there are two or three collisions happening at the same time...
	//if (num_cols == 2) {
	//	if (col_r * col_b) {
	//		if (f_r < f_b) {
	//			col_b = 0;
	//		}
	//		else {
	//			col_r = 0;
	//		}
	//	}

	//	if (col_r * col_f) {
	//		if (f_r < f_f) {
	//			col_f = 0;
	//		}
	//		else {
	//			col_r = 0;
	//		}
	//	}

	//	if (col_r * col_d) {
	//		if (f_r < f_d) {
	//			col_d = 0;
	//		}
	//		else {
	//			col_r = 0;
	//		}
	//	}

	//	if (col_r * col_u) {
	//		if (f_r < f_u) {
	//			col_u = 0;
	//		}
	//		else {
	//			col_r = 0;
	//		}

	//	}

	//	if (col_b * col_u) {
	//		if (f_b < f_u) {
	//			col_u = 0;
	//		}
	//		else {
	//			col_b = 0;
	//		}
	//	}

	//	if (col_b * col_d) {
	//		if (f_b < f_d) {
	//			col_d = 0;
	//		}
	//		else {
	//			col_b = 0;
	//		}
	//	}

	//	if (col_b * col_l) {
	//		if (f_b < f_l) {
	//			col_l = 0;
	//		}
	//		else {
	//			col_b = 0;
	//		}
	//	}

	//	if (col_l * col_u) {
	//		if (f_l < f_u) {
	//			col_u = 0;
	//		}
	//		else {
	//			col_l = 0;
	//		}
	//	}

	//	if (col_l * col_d) {
	//		if (f_l < f_d) {
	//			col_d = 0;
	//		}
	//		else {
	//			col_l = 0;
	//		}
	//	}

	//	if (col_l * col_f) {
	//		if (f_l < f_f) {
	//			col_f = 0;
	//		}
	//		else {
	//			col_l = 0;
	//		}
	//	}

	//	if (col_f * col_u) {
	//		if (f_f < f_u) {
	//			col_u = 0;
	//		}
	//		else {
	//			col_f = 0;
	//		}
	//	}

	//	if (col_f * col_d) {
	//		if (f_f < f_d) {
	//			col_d = 0;
	//		}
	//		else {
	//			col_f = 0;
	//		}
	//	}
	//}
	//else if (num_cols == 3) {

	//	if (col_r * col_f * col_d) {

	//		if (f_r < f_f) {
	//			col_f = 0;
	//			if (f_r < f_d) {
	//				col_d = 0;
	//			}
	//			else {
	//				col_r = 0;
	//			}
	//		}
	//		else {
	//			col_r = 0;
	//			if (f_f < f_d) {
	//				col_d = 0;
	//			}
	//			else {
	//				col_f = 0;
	//			}
	//		}

	//	}

	//	if (col_r * col_f * col_u) {

	//		if (f_r < f_f) {
	//			col_f = 0;
	//			if (f_r < f_u) {
	//				col_u = 0;
	//			}
	//			else {
	//				col_r = 0;
	//			}
	//		}else {
	//			col_r = 0;
	//			if (f_f < f_u) {
	//				col_u = 0;
	//			}
	//			else {
	//				col_f = 0;
	//			}
	//		}

	//	}

	//	if (col_r * col_b * col_d) {

	//		if (f_r < f_b) {
	//			col_b = 0;
	//			if (f_r < f_d) {
	//				col_d = 0;
	//			}
	//			else {
	//				col_r = 0;
	//			}
	//		}
	//		else {
	//			col_r = 0;
	//			if (f_b < f_d) {
	//				col_d = 0;
	//			}
	//			else {
	//				col_b = 0;
	//			}
	//		}


	//	}

	//	if (col_r * col_b * col_u) {
	//		if (f_r < f_b) {
	//			col_b = 0;
	//			if (f_r < f_u) {
	//				col_u = 0;
	//			}
	//			else {
	//				col_r = 0;
	//			}
	//		}
	//		else {
	//			col_r = 0;
	//			if (f_b < f_u) {
	//				col_u = 0;
	//			}
	//			else {
	//				col_b = 0;
	//			}
	//		}
	//	}

	//	if (col_l * col_f * col_d) {
	//		if (f_l < f_f) {
	//			col_f = 0;
	//			if (f_l < f_d) {
	//				col_d = 0;
	//			}
	//			else {
	//				col_l = 0;
	//			}
	//		}
	//		else {
	//			col_l = 0;
	//			if (f_f < f_d) {
	//				col_d = 0;
	//			}
	//			else {
	//				col_f = 0;
	//			}
	//		}
	//	}

	//	if (col_l * col_f * col_u) {
	//		if (f_l < f_f) {
	//			col_f = 0;
	//			if (f_l < f_u) {
	//				col_u = 0;
	//			}
	//			else {
	//				col_l = 0;
	//			}
	//		}
	//		else {
	//			col_l = 0;
	//			if (f_f < f_u) {
	//				col_u = 0;
	//			}
	//			else {
	//				col_f = 0;
	//			}
	//		}
	//	}

	//	if (col_l * col_b *col_d) {
	//		if (f_l < f_b) {
	//			col_b = 0;
	//			if (f_l < f_d) {
	//				col_d = 0;
	//			}
	//			else {
	//				col_l = 0;
	//			}
	//		}
	//		else {
	//			col_l = 0;
	//			if (f_b < f_d) {
	//				col_d = 0;
	//			}
	//			else {
	//				col_b = 0;
	//			}
	//		}
	//	}

	//	if (col_l * col_b * col_u) {

	//		if (f_l < f_b) {
	//			col_b = 0;
	//			if (f_l < f_u) {
	//				col_u = 0;
	//			}
	//			else {
	//				col_l = 0;
	//			}
	//		}
	//		else {
	//			col_l = 0;
	//			if (f_b < f_u) {
	//				col_u = 0;
	//			}
	//			else {
	//				col_b = 0;
	//			}
	//		}
	//	}
	//}
	//
//}

void Ball::init() {
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);

	glGenBuffers(1, &norBufID);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);

	//glGenBuffers(1, &texBufID);
	//glBindBuffer(GL_ARRAY_BUFFER, texBufID);
	//glBufferData(GL_ARRAY_BUFFER, texBuf.size() * sizeof(float), &texBuf[0], GL_STATIC_DRAW);

	glGenBuffers(1, &eleBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, eleBuf.size() * sizeof(unsigned int), &eleBuf[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	assert(glGetError() == GL_NO_ERROR);
};

void Ball::tare() {

};

void Ball::reset() {

	// To generate multiple different example cases running by allowing the user to have random starting conditions in the same framework
	// Just push "r" keyboard button 
	srand(time(0)); // This will ensure a really randomized number by help of time.

	int va = rand() % 40 - 21; // Randomizing the number between -20 - + 20.
	int vb = rand() % 40 - 21;
	int vc = rand() % 40 - 21;

	int xa = rand() % 8 - 4;
	int xb = rand() % 8 - 0;
	int xc = rand() % 8 - 4;

	Vector3d random_v;
	random_v << va, vb, vc;
	v = random_v;
	//center = center_0; // Initial position
	Vector3d random_x;
	random_x << xa, xb, xc;
	center = random_x;
	updatePosNor();
};

void Ball::draw(std::shared_ptr<MatrixStack> MV, const std::shared_ptr<Program> p, std::shared_ptr<Box> box)const {
	glUniform3fv(p->getUniform("kdFront"), 1, Vector3f(1.0, 0.0, 0.0).data());
	glUniform3fv(p->getUniform("kdBack"), 1, Vector3f(1.0, 1.0, 0.0).data());
	MV->pushMatrix();

	glUniformMatrix4fv(p->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));

	int h_pos = p->getAttribute("aPos");
	glEnableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size() * sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	
	//int h_nor = p->getAttribute("aNor");
	//glEnableVertexAttribArray(h_nor);
	//glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	//glBufferData(GL_ARRAY_BUFFER, norBuf.size() * sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glDrawElements(GL_QUADS, 4*rings*sectors, GL_UNSIGNED_INT, (const void *)0);
	//glDisableVertexAttribArray(h_nor);
	glDisableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	MV->popMatrix();

};

Ball::~Ball() {

}

void Ball::step_w(double h, const Vector3d &grav, const std::shared_ptr<Box> box) {

	/*if (abs(v[1]) <= 0.01&& center[1] - r< 0.1) {
	v[1] = 0;
	center[1] = r;
	updatePosNor();

	}
	else {*/
	double wind_coef = 2;
	Vector3d wind_dir(-1, 0, 0);

	double coef_air = 1;    // Air friction coefficient
	double epsilon = -0.95;   // Elascity  
	double mu = 0.1;		// Friction
	double hremaining = h;
	double timestep = hremaining;

	while (hremaining > 0.00001) {

		double fraction;
		Vector3d normal;
		// Euler Integration
		int num_cols = 0;
		Vector3d v_new = v + hremaining * (grav - coef_air * v + wind_coef * wind_dir);
		Vector3d center_new = center + hremaining * v_new; //

														   // Collision Detection for six faces

		int col_r = ((center_new[0] - box->right + r) > EPSILON);
		int col_l = ((center_new[0] - box->left - r) < -EPSILON);
		int col_u = ((center_new[1] - box->up + r) > EPSILON);
		int col_d = ((center_new[1] - box->down - r) < -EPSILON);
		int col_f = ((center_new[2] - box->forward + r) > EPSILON);
		int col_b = ((center_new[2] - box->back - r) < -EPSILON);

		/* Collision Detection: sign(dn)?=sign(dn+1) if it is opposite sign, then are at opposite plane

		int col_r = ((center_new[0] - box->right + r)* (center[0] - box->right + r) < -EPSILON);
		int col_l = ((center_new[0] - box->left - r)*(center[0] - box->left - r) < -EPSILON);
		int col_u = ((center_new[1] - box->up + r)*(center[1] - box->up + r) < -EPSILON);
		int col_d = ((center_new[1] - box->down - r)*(center[1] - box->down - r) < -EPSILON);
		int col_f = ((center_new[2] - box->forward + r)*(center[2] - box->forward + r) < -EPSILON);
		int col_b = ((center_new[2] - box->back - r)*(center[2] - box->back - r) < -EPSILON);
		*/

		// Determination 
		double f_r = (center[0] - box->right + r) / ((center[0] - box->right) - (center_new[0] - box->right));
		double f_l = (center[0] - box->left - r) / ((center[0] - box->left) - (center_new[0] - box->left));
		double f_u = (center[1] - box->up + r) / ((center[1] - box->up) - (center_new[1] - box->up));
		double f_d = (center[1] - box->down - r) / ((center[1] - box->down) - (center_new[1] - box->down));
		double f_f = (center[2] - box->forward + r) / ((center[2] - box->forward) - (center_new[2] - box->forward));
		double f_b = (center[2] - box->back - r) / ((center[2] - box->back) - (center_new[2] - box->back));


		num_cols = col_r + col_l + col_u + col_d + col_f + col_b;

		if (num_cols != 0) {

			if (col_r) {
				normal << -1, 0, 0;
				fraction = f_r;
			}
			if (col_l) {
				normal << 1, 0, 0;
				fraction = f_l;
			}
			if (col_u) {
				normal << 0, -1, 0;
				fraction = f_u;
			}
			if (col_d) {
				normal << 0, 1, 0;
				fraction = f_d;
			}
			if (col_f) {
				normal << 0, 0, -1;
				fraction = f_f;
			}
			if (col_b) {
				normal << 0, 0, 1;
				fraction = f_b;
			}

			// Response
			timestep = fraction * timestep;
			Vector3d vc = v + fraction * h * (grav - coef_air * v + wind_coef * wind_dir);
			Vector3d center_c = center + fraction * h * v;

			Vector3d vc_n = (vc.dot(normal))*normal;
			Vector3d vc_t = vc - vc_n;
			Vector3d v_new_n = epsilon * vc_n;
			Vector3d v_new_t = (1 - mu)* vc_t;

			v_new = v_new_n + v_new_t;
			center_new = center_c;
			v = v_new;
			center = center_new;
			hremaining = hremaining - timestep;
			updatePosNor();

		}
		else {
			// If there is no collision 
			hremaining = hremaining - timestep;
			v = v_new;
			//cout << v << endl << endl;
			center = center_new;
			//cout << center << endl << endl;
			updatePosNor();
		}
	}

	//updatePosNor();
}
//if (num_cols == 2) {
//	if (col_r * col_b) {
//		if (f_r < f_b) {
//			col_b = 0;
//		}
//		else {
//			col_r = 0;
//		}
//	}

//	if (col_r * col_f) {
//		if (f_r < f_f) {
//			col_f = 0;
//		}
//		else {
//			col_r = 0;
//		}
//	}

//	if (col_r * col_d) {
//		if (f_r < f_d) {
//			col_d = 0;
//		}
//		else {
//			col_r = 0;
//		}
//	}

//	if (col_r * col_u) {
//		if (f_r < f_u) {
//			col_u = 0;
//		}
//		else {
//			col_r = 0;
//		}

//	}

//	if (col_b * col_u) {
//		if (f_b < f_u) {
//			col_u = 0;
//		}
//		else {
//			col_b = 0;
//		}
//	}

//	if (col_b * col_d) {
//		if (f_b < f_d) {
//			col_d = 0;
//		}
//		else {
//			col_b = 0;
//		}
//	}

//	if (col_b * col_l) {
//		if (f_b < f_l) {
//			col_l = 0;
//		}
//		else {
//			col_b = 0;
//		}
//	}

//	if (col_l * col_u) {
//		if (f_l < f_u) {
//			col_u = 0;
//		}
//		else {
//			col_l = 0;
//		}
//	}

//	if (col_l * col_d) {
//		if (f_l < f_d) {
//			col_d = 0;
//		}
//		else {
//			col_l = 0;
//		}
//	}

//	if (col_l * col_f) {
//		if (f_l < f_f) {
//			col_f = 0;
//		}
//		else {
//			col_l = 0;
//		}
//	}

//	if (col_f * col_u) {
//		if (f_f < f_u) {
//			col_u = 0;
//		}
//		else {
//			col_f = 0;
//		}
//	}

//	if (col_f * col_d) {
//		if (f_f < f_d) {
//			col_d = 0;
//		}
//		else {
//			col_f = 0;
//		}
//	}
//}
//else if (num_cols == 3) {

//	if (col_r * col_f * col_d) {

//		if (f_r < f_f) {
//			col_f = 0;
//			if (f_r < f_d) {
//				col_d = 0;
//			}
//			else {
//				col_r = 0;
//			}
//		}
//		else {
//			col_r = 0;
//			if (f_f < f_d) {
//				col_d = 0;
//			}
//			else {
//				col_f = 0;
//			}
//		}

//	}

//	if (col_r * col_f * col_u) {

//		if (f_r < f_f) {
//			col_f = 0;
//			if (f_r < f_u) {
//				col_u = 0;
//			}
//			else {
//				col_r = 0;
//			}
//		}else {
//			col_r = 0;
//			if (f_f < f_u) {
//				col_u = 0;
//			}
//			else {
//				col_f = 0;
//			}
//		}

//	}

//	if (col_r * col_b * col_d) {

//		if (f_r < f_b) {
//			col_b = 0;
//			if (f_r < f_d) {
//				col_d = 0;
//			}
//			else {
//				col_r = 0;
//			}
//		}
//		else {
//			col_r = 0;
//			if (f_b < f_d) {
//				col_d = 0;
//			}
//			else {
//				col_b = 0;
//			}
//		}


//	}

//	if (col_r * col_b * col_u) {
//		if (f_r < f_b) {
//			col_b = 0;
//			if (f_r < f_u) {
//				col_u = 0;
//			}
//			else {
//				col_r = 0;
//			}
//		}
//		else {
//			col_r = 0;
//			if (f_b < f_u) {
//				col_u = 0;
//			}
//			else {
//				col_b = 0;
//			}
//		}
//	}

//	if (col_l * col_f * col_d) {
//		if (f_l < f_f) {
//			col_f = 0;
//			if (f_l < f_d) {
//				col_d = 0;
//			}
//			else {
//				col_l = 0;
//			}
//		}
//		else {
//			col_l = 0;
//			if (f_f < f_d) {
//				col_d = 0;
//			}
//			else {
//				col_f = 0;
//			}
//		}
//	}

//	if (col_l * col_f * col_u) {
//		if (f_l < f_f) {
//			col_f = 0;
//			if (f_l < f_u) {
//				col_u = 0;
//			}
//			else {
//				col_l = 0;
//			}
//		}
//		else {
//			col_l = 0;
//			if (f_f < f_u) {
//				col_u = 0;
//			}
//			else {
//				col_f = 0;
//			}
//		}
//	}

//	if (col_l * col_b *col_d) {
//		if (f_l < f_b) {
//			col_b = 0;
//			if (f_l < f_d) {
//				col_d = 0;
//			}
//			else {
//				col_l = 0;
//			}
//		}
//		else {
//			col_l = 0;
//			if (f_b < f_d) {
//				col_d = 0;
//			}
//			else {
//				col_b = 0;
//			}
//		}
//	}

//	if (col_l * col_b * col_u) {

//		if (f_l < f_b) {
//			col_b = 0;
//			if (f_l < f_u) {
//				col_u = 0;
//			}
//			else {
//				col_l = 0;
//			}
//		}
//		else {
//			col_l = 0;
//			if (f_b < f_u) {
//				col_u = 0;
//			}
//			else {
//				col_b = 0;
//			}
//		}

//	}

//}

//

//}
