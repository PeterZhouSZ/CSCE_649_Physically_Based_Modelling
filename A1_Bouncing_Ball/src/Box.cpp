#include <iostream>
#include "Box.h"

#include "Particle.h"
#include "Program.h"
#include "GLSL.h"
#include "MatrixStack.h"
#include "Spring.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

Box::Box(double f, double b, double r, double l, double u, double d)
{
	this->forward = f;
	this->back = b;
	this->right = r;
	this->left = l;
	this->up = u;
	this->down = d;
}


Box::~Box()
{
}
