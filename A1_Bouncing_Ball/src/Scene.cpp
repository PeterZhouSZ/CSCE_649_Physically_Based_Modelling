#include <iostream>

#include "Scene.h"
#include "Particle.h"
//#include "Cloth.h"
#include "Tetrahedron.h"
#include "Ball.h"
#include "Shape.h"
#include "Program.h"
#include "Box.h"

using namespace std;
using namespace Eigen;

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
	h = 1e-3; // Change the time step
	
	grav << 0.0, -9.8, 0.0;

	//int rows = 2;
	//int cols = 2;
	double m = 1;
	double radius = 0.5;
	//double stiffness = 1e2;
	//Vector2d damping(0.0, 1.0);
	//Vector3d x00(-0.25, 0.5, 0.0);
	//Vector3d x01(0.25, 0.5, 0.0);
	//Vector3d x10(-0.25, 0.5, -0.5);
	//Vector3d x11(0.25, 0.5, -0.5);

	Vector3d x0(3,6, 2);
	Vector3d v0(20, 0, 0); // init the velocity

	//cloth = make_shared<Cloth>(rows, cols, x00, x01, x10, x11, mass, stiffness, damping);
	//tet = make_shared<Tetrahedron>(x0, x1, x2, x3, mass, stiffness, damping);
	
	ball = make_shared<Ball>(x0, v0, m, radius);

	double front = 5.0;
	double back = -5.0;
	double right = 5.0;
	double left = -5.0;
	double up = 10.0;
	double down = 0.0;

	box = make_shared<Box>(front, back, right, left, up, down);
	sphereShape = make_shared<Shape>();
	sphereShape->loadMesh(RESOURCE_DIR + "sphere2.obj");
	
	auto sphere = make_shared<Particle>(sphereShape);
	spheres.push_back(sphere);
	sphere->r = 0.1;
	sphere->x = Vector3d(0.0, 0.2, 0.0);
}

void Scene::init()
{
	sphereShape->init();
	//cloth->init();
	//tet->init();
	ball->init();
}

void Scene::tare()
{
	for(int i = 0; i < (int)spheres.size(); ++i) {
		spheres[i]->tare();
	}
	//cloth->tare();
	//tet->tare();
	ball->tare();
}

void Scene::reset()
{
	t = 0.0;
	for(int i = 0; i < (int)spheres.size(); ++i) {
		spheres[i]->reset();
	}
	//cloth->reset();
	//tet->reset();
	ball->reset();
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
	//cloth->step(h, grav, spheres);
	//tet->step(h, grav);
	//ball->step(h, grav, box);
	ball->step_poly(h, grav,box);
}

void Scene::step_w()
{
	t += h;
	// Simulate the cloth
	//cloth->step(h, grav, spheres);
	//tet->step(h, grav);
	//ball->step(h, grav, box);
	ball->step_w(h, grav,box);

}



void Scene::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> prog) const
{
	glUniform3fv(prog->getUniform("kdFront"), 1, Vector3f(1.0, 1.0, 1.0).data());
	for(int i = 0; i < (int)spheres.size(); ++i) {
		//spheres[i]->draw(MV, prog);
	}
	//cloth->draw(MV, prog);
	//tet->draw(MV, prog);
	ball->draw(MV, prog,box);
}
