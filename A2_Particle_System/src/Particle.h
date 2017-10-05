#pragma once
#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <memory>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>

#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>
#include <tetgen.h>
class MatrixStack;
class Program;
class Texture;

class Particle
{
public:
	Particle(int index);
	virtual ~Particle();
	void rebirth(float t, const bool *keyToggles);
	void step(float t, float h, const Eigen::Vector3f &g, const bool *keyToggles);
	
	// Static, shared by all particles
	static void init(int n);
	static void draw(const std::vector< std::shared_ptr<Particle> > &particles,
					 std::shared_ptr<Program> prog);
	static float randFloat(float l, float h);
	
private:
	// Properties that are fixed
	Eigen::Map<Eigen::Vector3f> color; // color (mapped to a location in colBuf)
	float &scale;                      // size (mapped to a location in scaBuf)
	int idx;
	
	// Properties that changes every rebirth
	float m;        // mass
	float d;        // viscous damping
	float lifespan; // how long this particle lives
	float tEnd;     // time this particle dies
	
	// Properties that changes every frame
	Eigen::Map<Eigen::Vector3f> x; // position (mapped to a location in posBuf)
	Eigen::Vector3f v;             // velocity
	float &alpha;                  // mapped to a location in alpBuf
	
	// Static, shared by all particles
	static std::vector<float> posBuf;
	static std::vector<float> colBuf;
	static std::vector<float> alpBuf;
	static std::vector<float> scaBuf;
	static GLuint posBufID;
	static GLuint colBufID;
	static GLuint alpBufID;
	static GLuint scaBufID;

};

#endif
