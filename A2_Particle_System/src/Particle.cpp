#include "Particle.h"
#include <iostream>

#include "GLSL.h"
#include "MatrixStack.h"
#include "Program.h"
#include "Texture.h"

using namespace std;
using namespace Eigen;

vector<float> Particle::posBuf;
vector<float> Particle::colBuf;
vector<float> Particle::alpBuf;
vector<float> Particle::scaBuf;

GLuint Particle::posBufID;
GLuint Particle::colBufID;
GLuint Particle::alpBufID;
GLuint Particle::scaBufID;



// This test function is adapted from Moller-Trumbore intersection algorithm. 
// See https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm 

bool rayTriangleIntersects(Vector3f v1, Vector3f v2, Vector3f v3, Vector3f dir, Vector3f pos) {

	Vector3f e1 = v2 - v1;
	Vector3f e2 = v3 - v1;

	// Calculate planes normal vector
	//cross product
	Vector3f pvec = dir.cross(e2);

	//dot product
	float det = e1.dot(pvec);

	// Ray is parallel to plane
	if (det <1e-8 && det > -1e-8) {
		return false;
	}

	float inv_det = 1 / det;

	// Distance from v1 to ray pos
	Vector3f tvec = pos - v1;
	float u = (tvec.dot(pvec))*inv_det;
	if (u < 0 || u > 1) {
		return false;
	}

	Vector3f qvec = tvec.cross(e1);
	float v = dir.dot(qvec) * inv_det;
	if (v<0 || u + v>1) {
		return false;
	}

	float t = e2.dot(qvec) * inv_det;
	if (t > 1e-8) return true;
	return false;
}


// Before this constructor is called, posBuf must be a valid vector<float>.
// I.e., Particle::init(n) must be called first.
Particle::Particle(int index) :
	color(&colBuf[3*index]),
	scale(scaBuf[index]),
	x(&posBuf[3*index]),
	alpha(alpBuf[index])
{
	idx = index;
	//in_2.load_off("N");
	//tetrahedralize("pqz", &in_2, &out);

	// Random fixed properties
	color << randFloat(0.5f, 1.0f), randFloat(0.5f, 1.0f), randFloat(0.5f, 1.0f);
	scale = randFloat(0.1f, 0.2f);
	
	// Send color data to GPU
	glBindBuffer(GL_ARRAY_BUFFER, colBufID);
	glBufferSubData(GL_ARRAY_BUFFER, 3*index*sizeof(float), 3*sizeof(float), color.data());
	
	// Send scale data to GPU
	glBindBuffer(GL_ARRAY_BUFFER, scaBufID);
	glBufferSubData(GL_ARRAY_BUFFER, index*sizeof(float), sizeof(float), &scale);
}

Particle::~Particle()
{
}

void Particle::rebirth(float t, const bool *keyToggles)
{
	m = 1.0f;
	alpha = 1.0f;
	x << randFloat(-10.0f, 10.0f), randFloat(-10.0f, 10.0f), randFloat(-10.0f, 10.0f);
	
	if(keyToggles[(unsigned)'g']) {
		// Gravity towards origin
		color << randFloat(0.5f, 1.0f), randFloat(0.5f, 1.0f), randFloat(0.5f, 1.0f);
		// Send color data to GPU
		glBindBuffer(GL_ARRAY_BUFFER, colBufID);
		glBufferSubData(GL_ARRAY_BUFFER, 3 * idx * sizeof(float), 3 * sizeof(float), color.data());
		d = randFloat(0.0f, 0.0f);
		x << randFloat(-3.0f, 3.0f), randFloat(-3.0f, 3.0f), randFloat(-3.0f, 3.0f);
		v << randFloat(-1.0f, 1.0f), randFloat(-1.0f, 1.0f), randFloat(-1.0f, 1.0f);
		lifespan = 1.0f;
	} else {
		// Gravity downwards
		color << randFloat(245.0/255.0f, 1.0f), randFloat(235.0/255.0f, 1.0f), randFloat(200.0/256.0f, 1.0f);
		// Send color data to GPU
		glBindBuffer(GL_ARRAY_BUFFER, colBufID);
		glBufferSubData(GL_ARRAY_BUFFER, 3 * idx * sizeof(float), 3 * sizeof(float), color.data());

		d = randFloat(0.0f, 0.0f);
		x << randFloat(-5.0f, 5.0f), randFloat(-5.0f, 5.0f), randFloat(-5.0f, 5.0f);	
		v << randFloat(-1.0f, 1.0f), randFloat(-1.0f, 1.0f), randFloat(-1.0f, 1.0f);
		lifespan = 1.0f;
	}

	if (keyToggles[(unsigned)'p']) {

		// Gravity downwards
		color << randFloat(245.0 / 255.0f, 1.0f), randFloat(235.0 / 255.0f, 1.0f), randFloat(200.0 / 256.0f, 1.0f);
		// Send color data to GPU
		glBindBuffer(GL_ARRAY_BUFFER, colBufID);
		glBufferSubData(GL_ARRAY_BUFFER, 3 * idx * sizeof(float), 3 * sizeof(float), color.data());

		d = randFloat(0.0f, 0.0f);
		x << randFloat(-2.0f, 2.0f), randFloat(0.0f, 1.0f), randFloat(-2.0f, 2.0f);
		// x << randFloat(-0.1f, 0.1f), randFloat(-0.1f, 0.1f), randFloat(-0.1f, 0.1f);
		v << randFloat(-1.0f, 1.0f), randFloat(-1.0f, 1.0f), randFloat(-1.0f, 1.0f);
		lifespan = 2.5f;

	}
	
	if (keyToggles[(unsigned)'a']) {
		// Generate from a certain point to different directions, like fireworks
		// Red and Orange
		color << 255.0, randFloat(0.0f, 1.0f), randFloat(0.0f, 105.0 / 255.0f);
		// Send color data to GPU
		glBindBuffer(GL_ARRAY_BUFFER, colBufID);
		glBufferSubData(GL_ARRAY_BUFFER, 3 * idx * sizeof(float), 3 * sizeof(float), color.data());

		d = randFloat(0.0f, 0.0f);
		x << 0.0, 1.0, 0.0;
		v << randFloat(-5.0f, 5.0f), randFloat(-7.0f, 7.0f), randFloat(-6.0f, 6.0f);
		lifespan = 1.0f;
	}

	if (keyToggles[(unsigned)'b']) {
		// Generate from a point towards a certain direction, 
		color << randFloat(0.0f, 60.0/255.0f), randFloat(100.0/255.0f, 224.0/256.0f), randFloat(70.0/255.0f, 255.0 / 255.0f);

		// Send color data to GPU
		glBindBuffer(GL_ARRAY_BUFFER, colBufID);
		glBufferSubData(GL_ARRAY_BUFFER, 3 * idx * sizeof(float), 3 * sizeof(float), color.data());

		d = randFloat(0.0f, 0.0f);
		x << 0.0, 0.0, 0.0;
		v << randFloat(-0.1f, 0.1f), randFloat(-0.1f, 0.1f), randFloat(-0.1f, 0.1f);
		Vector3f dir(0.0f, 1.0f, 0.0f);
		v = (v + dir)*7.0;
		lifespan = 1.0f;
	}

	if (keyToggles[(unsigned)'t']) {
		// Generate from a triangle towards the normal direction
		color << randFloat(0.0f, 60.0 / 255.0f), randFloat(100.0 / 255.0f, 224.0 / 256.0f), randFloat(70.0 / 255.0f, 255.0 / 255.0f);
		// Send color data to GPU
		glBindBuffer(GL_ARRAY_BUFFER, colBufID);
		glBufferSubData(GL_ARRAY_BUFFER, 3 * idx * sizeof(float), 3 * sizeof(float), color.data());

		d = randFloat(0.0f, 0.0f);
		float u = randFloat(0.0f, 1.0f);
		float v1 = randFloat(0.0f, 1.0-u);
		float w = 1 - u - v1;
		Vector3f p1(sin(1 / 6.0 * 2 * 3.14159), 0.0f, cos(1 / 6.0 * 2 * 3.14159));
		Vector3f p2(sin(3 / 6.0 * 2 * 3.14159), 0.0f, cos(3 / 6.0 * 2 * 3.14159));
		Vector3f p3(sin(5 / 6.0 * 2 * 3.14159), 0.0f, cos(5 / 6.0 * 2 * 3.14159));
		x = u * p1 + v1 * p2 + w * p3;
		v << 0.0f, 6.0f, 0.0f;
		lifespan = 1.0f;
	}

	if (keyToggles[(unsigned)'d']) {
		// Generate from a circle towards the normal direction
		color << randFloat(0.0f, 60.0 / 255.0f), randFloat(100.0 / 255.0f, 224.0 / 256.0f), randFloat(70.0 / 255.0f, 255.0 / 255.0f);

		// Send color data to GPU
		glBindBuffer(GL_ARRAY_BUFFER, colBufID);
		glBufferSubData(GL_ARRAY_BUFFER, 3 * idx * sizeof(float), 3 * sizeof(float), color.data());

		d = randFloat(0.0f, 0.0f);
		float u = randFloat(0.0f, 1.0f);
		float w = randFloat(0.0f, 2*3.14159f);
		x << (1 - u*u)*1.5*cos(w), 0.0f, (1 - u*u)*1.5*sin(w);
		v << 0.0f, 6.0f, 0.0f;
		lifespan = 1.0f;
	}
	tEnd = t + lifespan;
}

void Particle::step(float t, float h, const Vector3f &g, const bool *keyToggles)
{
	if(t > tEnd) {
		rebirth(t, keyToggles);
	}
	// Update alpha based on current time
	alpha = (tEnd-t)/lifespan;

	Vector3f f;

	if(keyToggles[(unsigned)'g']) {
		// Gravity towards origin
		f = -15 * m/((int)(x.dot(x) + 0.01 * 0.01)^(3/2)) * x;
	} else {
		// Gravity downwards
		f = m * g -1 * v;
	}
	
	if (keyToggles[(unsigned)'y']) {
		// Potential field with a sphere centered in the origin
		Vector3f center(0.0f, 0.0f, 0.0f);
		f -= m * 10 * (1 / (0.5F - (x - center).norm()))*(x - center)/(x-center).norm();
	}

	if (keyToggles[(unsigned)'o']) {
		// Wind
		Vector3f wind(10.0f, 0.0f, 0.0f);
		f += wind;
	}
	
	Vector3f v_old = v;
	Vector3f x_old = x;
	v += h*f/m;
	
	Vector3f v_new = v;
	
	// Update position
	x += h*v;
	Vector3f x_new = x ;

	// Apply floor collision
	if(keyToggles[(unsigned)'f']) {
		
		if(x(1)<=0.0){
			v(1)=-v(1);
			x(1)=0.0;
		}	
	}

	// Apply polygon collision
	if (keyToggles[(unsigned)'p']) {
		Vector3f dir = x_new - x_old;
		Vector3f v1, v2, v3, v4, v5, v6;
		v1 << sin(1 / 6.0 * 2 * 3.14159), 0.0f,
			cos(1 / 6.0 * 2 * 3.14159);
		v2 << sin(3 / 6.0 * 2 * 3.14159), 0.0f,
			cos(3 / 6.0 * 2 * 3.14159);
		v3 << sin(5 / 6.0 * 2 * 3.14159), 0.0f,
			cos(5 / 6.0 * 2 * 3.14159);
		Vector3f normal = -(v1 - v3).cross(v2 - v3);
		float twoA = normal.norm();
		Vector3f nor = normal / twoA;
		float f= (x_old - v1).dot(nor) / ((x_old - v1).dot(nor) - (x_new - v1).dot(nor));

		v4 << sin(0 / 6.0 * 2 * 3.14159), 0.0f,
			cos(0 / 6.0 * 2 * 3.14159);
		v5 << sin(2 / 6.0 * 2 * 3.14159), 0.0f,
			cos(2 / 6.0 * 2 * 3.14159);
		v6 << sin(4 / 6.0 * 2 * 3.14159), 0.0f,
			cos(4 / 6.0 * 2 * 3.14159);
		Vector3f normal2 = -(v4 - v6).cross(v5 - v6);
		float twoA2 = normal2.norm();
		Vector3f nor2 = normal2 / twoA2;
		float f2 = (x_old - v4).dot(nor2) / ((x_old - v4).dot(nor2) - (x_new - v4).dot(nor2));

		if (rayTriangleIntersects(v1, v2, v3, dir, x_old)&& f<1.0) {
			
			Vector3f x_c = x_old + f * h * v_old;
			Vector3f vc_n = (v_new.dot(nor))*nor;
			Vector3f vc_t = v_new - vc_n;
			Vector3f v_new_n = -0.9 * vc_n;
			Vector3f v_new_t = 0.8 * vc_t;
			Vector3f v_neww = v_new_n + v_new_t;
			x = x_c;
			v = v_neww;			
		}
		else if (rayTriangleIntersects(v4, v5, v6, dir, x_old) && f2 < 1.0) {

			Vector3f x_c = x_old + f2 * h * v_old;
			Vector3f vc_n = (v_new.dot(nor2))*nor2;
			Vector3f vc_t = v_new - vc_n;
			Vector3f v_new_n = -0.9 * vc_n;
			Vector3f v_new_t = 0.8 * vc_t;
			Vector3f v_neww = v_new_n + v_new_t;
			x = x_c;
			v = v_neww;

		}else {
			v = v_new;
			x = x_new;
		}
	}

	if (keyToggles[(unsigned)'v']) {

		// Apply vortex 
		Vector3f x_vot(0.0f, 1.0f, 0.0f);
		Vector3f dx = x_old - x_vot;
		Vector3f vv(-dx(2)*1.0, v(1) ,dx(0)*1.0);
		float factor = 2.0 / (1+(dx.norm()/1.0));
		v += (vv - v)*factor;
		x = x_new;
	}
}


float Particle::randFloat(float l, float h)
{
	float r = rand() / (float)RAND_MAX;
	return (1.0f - r) * l + r * h;
}

void Particle::init(int n)
{
	posBuf.resize(3*n);
	colBuf.resize(3*n);
	alpBuf.resize(n);
	scaBuf.resize(n);
	
	for(int i = 0; i < n; ++i) {
		posBuf[3*i+0] = 0.0f;
		posBuf[3*i+1] = 0.0f;
		posBuf[3*i+2] = 0.0f;
		colBuf[3*i+0] = 1.0f;
		colBuf[3*i+1] = 1.0f;
		colBuf[3*i+2] = 1.0f;
		alpBuf[i] = 1.0f;
		scaBuf[i] = 1.0f;
	}

	// Generate buffer IDs
	GLuint bufs[4];
	glGenBuffers(4, bufs);
	posBufID = bufs[0];
	colBufID = bufs[1];
	alpBufID = bufs[2];
	scaBufID = bufs[3];
	
	// Send color buffer to GPU
	glBindBuffer(GL_ARRAY_BUFFER, colBufID);
	glBufferData(GL_ARRAY_BUFFER, colBuf.size()*sizeof(float), &colBuf[0], GL_STATIC_DRAW);
	
	// Send scale buffer to GPU
	glBindBuffer(GL_ARRAY_BUFFER, scaBufID);
	glBufferData(GL_ARRAY_BUFFER, scaBuf.size()*sizeof(float), &scaBuf[0], GL_STATIC_DRAW);
	
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	

	assert(glGetError() == GL_NO_ERROR);
}

void Particle::draw(const vector< shared_ptr<Particle> > &particles,
					shared_ptr<Program> prog)
{
	// Enable, bind, and send position array
	glEnableVertexAttribArray(prog->getAttribute("aPos"));
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(prog->getAttribute("aPos"), 3, GL_FLOAT, GL_FALSE, 0, 0);
	
	// Enable, bind, and send alpha array
	glEnableVertexAttribArray(prog->getAttribute("aAlp"));
	glBindBuffer(GL_ARRAY_BUFFER, alpBufID);
	glBufferData(GL_ARRAY_BUFFER, alpBuf.size()*sizeof(float), &alpBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(prog->getAttribute("aAlp"), 1, GL_FLOAT, GL_FALSE, 0, 0);
	
	// Enable and bind color array
	glEnableVertexAttribArray(prog->getAttribute("aCol"));
	glBindBuffer(GL_ARRAY_BUFFER, colBufID);
	glVertexAttribPointer(prog->getAttribute("aCol"), 3, GL_FLOAT, GL_FALSE, 0, 0);
	
	// Enable and bind scale array
	glEnableVertexAttribArray(prog->getAttribute("aSca"));
	glBindBuffer(GL_ARRAY_BUFFER, scaBufID);
	glVertexAttribPointer(prog->getAttribute("aSca"), 1, GL_FLOAT, GL_FALSE, 0, 0);
	
	// Draw
	glDrawArrays(GL_POINTS, 0, 3*particles.size());
	
	// Disable and unbind
	glDisableVertexAttribArray(prog->getAttribute("aSca"));
	glDisableVertexAttribArray(prog->getAttribute("aCol"));
	glDisableVertexAttribArray(prog->getAttribute("aAlp"));
	glDisableVertexAttribArray(prog->getAttribute("aPos"));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
}
