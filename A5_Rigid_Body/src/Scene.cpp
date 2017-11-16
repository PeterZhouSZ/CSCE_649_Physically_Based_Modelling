#include <iostream>

#include "Scene.h"
#include "Particle.h"
#include "Cloth.h"
#include "Tetrahedron.h"
//#include "Bar.h"
#include "Shape.h"
#include "Program.h"
#include "FemTet.h"
#include "FemNesme.h"
#include "FemSimit.h"
#include "RigidBody.h"

using namespace std;
using namespace Eigen;

#define SCENE 5
// 1: 2: Vert to Face 3: Edge to Edge 4: Rabit 5: five objs bouncing

// 1: LINEAR 2: STVK 3: NEOHOOKEAN 4: COROTATED 6: Simit 7: Nesme
#define MODE 1
// 10: FemTet 11: FemNesme 12: FemSimit 
// 20: RigidBody
#define MODEL 20

Scene::Scene() :
	t(0.0),
	h(1e-1),
	grav(0.0, 0.0, 0.0)
{
}

Scene::~Scene()
{
}

void Scene::load(const string &RESOURCE_DIR)
{
	
	// Units: meters, kilograms, seconds
	if (MODE == 1) {
		h = 1e-2;
	}
	else if (MODE == 2) {
		h = 1e-3;
	}
	else if (MODE == 3) {
		h = 1e-3;
	}
	else if (MODE == 4) {
		h = 1e-4;
	}
	else if (MODE == 5) {
		h = 1e-2;
	}
	
	grav << 0.0, -10, 0.0;
	double mass = 166.667;
	double density = 0.1;
	double height = 12.0;
	
	Vector2d damping(0.8, 0.8);

	if (MODEL == 10) {
		femtet = make_shared<FemTet>(density, damping);
	}
	if (MODEL == 11) {
		h = 1e-2;
		femNesme = make_shared<FemNesme>(density, damping);
	}
	if (MODEL == 12) {
		h = 1e-3;
		femSimit = make_shared<FemSimit>(density, damping);
	}
	if (MODEL == 13) {
		int rows = 2;
		int cols = 2;
		double stiffness = 1e2;
		Vector3d x00(-0.25, 0.5, 0.0);
		Vector3d x01(0.25, 0.5, 0.0);
		Vector3d x10(-0.25, 0.5, -0.5);
		Vector3d x11(0.25, 0.5, -0.5);
		Vector3d x0(0.0, 0.95, -0.05);
		Vector3d x1(-0.1, 0.85, -0.05);
		Vector3d x2(0.0, 0.9, 0.0);
		Vector3d x3(0.1, 0.85, -0.05);
		cloth = make_shared<Cloth>(rows, cols, x00, x01, x10, x11, mass, stiffness, damping);
		sphereShape = make_shared<Shape>();
		sphereShape->loadMesh(RESOURCE_DIR + "sphere2.obj");
		auto sphere = make_shared<Particle>(sphereShape);
		spheres.push_back(sphere);
		sphere->r = 0.1;
		sphere->x = Vector3d(0.0, 0.2, 0.0);
	}
	if (MODEL == 14) {
		Vector3d x0(0.0, 5.0, 1.0);
		Vector3d x1(0.0, 5.0, 0.0);
		Vector3d x2(1.0, 5.0, 0.0);
		Vector3d x3(0.0, 6.0, 0.0);
		double stiffness = 1e2;
		tet = make_shared<Tetrahedron>(x0, x1, x2, x3, mass, stiffness, damping);

	}
	if (MODEL == 15) {
		Vector3d x0(0.0, 5.0, 1.0);
		Vector3d x1(0.0, 5.0, 0.0);
		Vector3d x2(1.0, 5.0, 0.0);
		Vector3d x3(0.0, 6.0, 0.0);
		//bar = make_shared<Bar>(x0, x1, x2, x3, density, height, damping);
	}

	if (MODEL == 20) {
		numrbs = 5;
		VectorXd posVec(3*numrbs);
		VectorXd vVec(3 * numrbs);
		VectorXd wVec(3 * numrbs);

		posVec <<0.0, 4.0, 0.0,
				0.0, -3.0, 0.0,
				-2.0, 3.5, 0.4,
				-3.5, -2.0, 1.0,
				-7.0, 7.0, 4.0;
		
		vVec << 0.0, 0.0, 0.0,
				0.0, 1.0, 0.0,
				0.0, 0.0, -0.4,
				0.0, -4.0, 0.0,
				3.0, 0.0, 0.0;
		
		wVec << 0.0, 0.0, 0.0,
				0.0, 0.0, 0.0,
				0.0, 0.0, 1.0,
				0.0, 0.0, 0.0,
				3.0, 0.0, 0.0;

		if (SCENE == 2) {
			numrbs = 2;
			for (int i = 0; i < numrbs; i++) {
				Vector3d x = posVec.segment<3>(i * 3);
				Vector3d v = vVec.segment<3>(i * 3);
				Vector3d w = wVec.segment<3>(i * 3);
				auto rb = make_shared<RigidBody>(i, i%2, 1, x, v, w);
				rigidbodies.push_back(rb);
			}
		}
		if (SCENE == 3) {
			numrbs = 2;

			for (int i = 0; i < numrbs; i++) {
				Vector3d x = posVec.segment<3>(i * 3);
				Vector3d v = vVec.segment<3>(i * 3);
				Vector3d w = wVec.segment<3>(i * 3);
				auto rb = make_shared<RigidBody>(i, 1, 1, x, v, w);
				rigidbodies.push_back(rb);
			}
		}
		
		if (SCENE == 4) {
			numrbs = 1;
			for (int i = 0; i < numrbs; i++) {
				Vector3d x = posVec.segment<3>(i * 3);
				Vector3d v = Vector3d(4.0, 2.0,0.0);
				Vector3d w = wVec.segment<3>(i * 3);
				auto rb = make_shared<RigidBody>(i, 0, 2, x, v, w);
				rigidbodies.push_back(rb);
			}
		}

		if (SCENE == 5) {
			numrbs = 5;
			for (int i = 0; i < numrbs; i++) {
				Vector3d x = posVec.segment<3>(i * 3);
				Vector3d v = vVec.segment<3>(i * 3);
				Vector3d w = wVec.segment<3>(i * 3);

				auto rb = make_shared<RigidBody>(i, i%2, 4, x, v, w);
				rigidbodies.push_back(rb);
			}
		}
	}
}

void Scene::init()
{
	//sphereShape->init();
	//cloth->init();
	//tet->init();
	//bar->init();
	if (MODEL == 10) {
		femtet->init();
	}
	if (MODEL == 11) {
		femNesme->init();
	}
	if (MODEL == 12) {
		femSimit->init();
	}
	if (MODEL == 20) {
		//rigidbody->init();
		for (int i = 0; i < numrbs; i++) {
			rigidbodies[i]->init();
		}
	}
}

void Scene::tare()
{
	/*for(int i = 0; i < (int)spheres.size(); ++i) {
		spheres[i]->tare();
	}*/
	//cloth->tare();
	//tet->tare();
	//bar->tare();
}

void Scene::reset()
{
	t = 0.0;
	/*for(int i = 0; i < (int)spheres.size(); ++i) {
		spheres[i]->reset();
	}*/
	//cloth->reset();
	//tet->reset();
	//bar->reset();
	//femtet->reset();
	//femNesme->reset();

}

void Scene::step()
{
	t += h;	
	// Move the sphere
	/*if(!spheres.empty()) {
		auto s = spheres.front();
		Vector3d x0 = s->x;
		double radius = 0.5;
		double a = 2.0*t;
		s->x(2) = radius * sin(a);
		Vector3d dx = s->x - x0;
		s->v = dx/h;
	}*/
	
	// Simulate the cloth
	//cloth->step(h, grav, spheres);
	//tet->step(h, grav);
	//bar->step(h, grav);

	if (MODEL == 10) {
		femtet->step(h, grav);
	}
	if (MODEL == 11) {
		femNesme->step(h, grav);
	}
	if (MODEL == 12) {
		femSimit->step(h, grav);
	}
	if (MODEL == 20) {
		//rigidbody->step(h, grav);
		if (SCENE == 2 ) {
			for (int i = 0; i < numrbs; i++) {
				rigidbodies[i]->step(h, grav, rigidbodies, 0);
			}
		}
		
		if (SCENE == 3) {
			for (int i = 0; i < numrbs; i++) {
				rigidbodies[i]->step(h, grav, rigidbodies, 1);
			}
		}

		if (SCENE == 4) {
			h = 1e-2;
			for (int i = 0; i < numrbs; i++) {
				rigidbodies[i]->step(h, grav, rigidbodies, 1);
			}
		}

		if (SCENE == 5) {
			h = 1e-2;
			for (int i = 0; i < numrbs; i++) {
				rigidbodies[i]->step(h, grav, rigidbodies, 1);
			}
		}
	}
}

void Scene::draw(shared_ptr<MatrixStack> MV, const shared_ptr<Program> prog) const
{
	glUniform3fv(prog->getUniform("kdFront"), 1, Vector3f(1.0, 1.0, 1.0).data());
	//for(int i = 0; i < (int)spheres.size(); ++i) {
	//	//spheres[i]->draw(MV, prog);
	//}
	//cloth->draw(MV, prog);
	//tet->draw(MV, prog);
	//bar->draw(MV, prog);
	//femtet->draw(MV, prog);
	//femNesme->draw(MV, prog);

	if (MODEL == 10) {
		femtet->draw(MV, prog);
	}
	if (MODEL == 11) {
		femNesme->draw(MV, prog);
	}
	if (MODEL == 12) {
		femSimit->draw(MV, prog);
	}
	if (MODEL == 20) {
		//rigidbody->draw(MV, prog);
		for (int i = 0; i < numrbs; i++) {
			rigidbodies[i]->draw(MV, prog);
		}
	}
}

double Scene::randDouble(double l, double h)
{
	float r = rand() / (double)RAND_MAX;
	return (1.0 - r) * l + r * h;
}