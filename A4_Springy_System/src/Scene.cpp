#include <iostream>
#include "Scene.h"
#include "Particle.h"
#include "Cloth.h"
#include "Shape.h"
#include "Program.h"
#include "Texture.h"

using namespace std;
using namespace Eigen;

// DEMO1: Euler vs. RK4
// DEMO2: Collision with sphere
// Demo3: Collision with face

#define DEMO 1

Scene::Scene() :
	t(0.0),
	h(1e-2),
	grav(0.0, 0.0, 0.0)
{
}

Scene::~Scene()
{
}

void Scene::load(const string &RESOURCE_DIR)
{
	// Units: meters, kilograms, seconds
	grav << 0.0, -9.8, 0.0;
	int rows = 10;
	int cols = 10;
	double stiffness = 1e2;
	h = 1e-2;

	// DEMO: Euler vs. RK4
	if (DEMO == 1) {
		h = 1e-3;
		rows = 2;
		cols = 2;
		stiffness = 1e1;
	}

	// DEMO: Collision with sphere
	if (DEMO == 2) {
		h = 1e-3;
		rows = 10;
		cols = 10;
		stiffness = 1e2;
	}
	
	if (DEMO == 4) {
		rows = 10;
		cols = 10;
		stiffness = 1e2;
		h = 1e-3;
	}

	double mass = 0.1;
	Vector2d damping(0.0, 1.0);
	Vector3d x00(-0.25, 0.5, 0.0);
	Vector3d x01(0.25, 0.5, 0.0);
	Vector3d x10(-0.25, 0.5, -0.5);
	Vector3d x11(0.25, 0.5, -0.5);

	cloth = make_shared<Cloth>(rows, cols, x00, x01, x10, x11, mass, stiffness, damping);

	// Init a sphere
	sphereShape = make_shared<Shape>();
	sphereShape->loadMesh(RESOURCE_DIR + "sphere2.obj");
	
	auto sphere = make_shared<Particle>(sphereShape);
	spheres.push_back(sphere);
	sphere->r = 0.1;
	sphere->x = Vector3d(0.0, 0.2, 0.0);

	// Init the texture of cloth and sphere
	texture_cloth = make_shared<Texture>();
	texture_cloth -> setFilename(RESOURCE_DIR + "tamu.jpg");

	texture_sphere = make_shared<Texture>();
	texture_sphere -> setFilename(RESOURCE_DIR + "earthKd.jpg");
}

void Scene::init()
{
	sphereShape->init();
	cloth->init();
	texture_cloth->init();
	texture_cloth->setUnit(0);
	texture_cloth->setWrapModes(GL_REPEAT, GL_REPEAT);

	texture_sphere->init();
	texture_sphere->setUnit(0);
	texture_sphere->setWrapModes(GL_REPEAT, GL_REPEAT);
}

void Scene::tare()
{
	for(int i = 0; i < (int)spheres.size(); ++i) {
		spheres[i]->tare();
	}
	cloth->tare();
}

void Scene::reset()
{
	t = 0.0;
	for(int i = 0; i < (int)spheres.size(); ++i) {
		spheres[i]->reset();
	}
	cloth->reset();
}

void Scene::step()
{
	t += h;
	
	// Move the sphere
	if(!spheres.empty()) {
		auto s = spheres.front();
		Vector3d x0 = s->x;
		double radius = 0.5;
		double a = 2.0*t;
		s->x(2) = radius * sin(a);
		Vector3d dx = s->x - x0;
		s->v = dx/h;
	}
	
	// Simulate the cloth
	cloth->step(h, grav, spheres);
}

void Scene::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> prog) const
{
	glUniform3fv(prog->getUniform("kdFront"), 1, Vector3f(1.0, 1.0, 1.0).data());
	texture_sphere->bind(prog->getUniform("texture_cloth_1"));
	for(int i = 0; i < (int)spheres.size(); ++i) {
		 spheres[i]->draw(MV, prog);
	}
	texture_sphere->unbind();
	texture_cloth->bind(prog->getUniform("texture_cloth_1"));
	cloth->draw(MV, prog);
	texture_cloth->unbind();
}
