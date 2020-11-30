#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <string>


class Particle {
public:
	using FV = Eigen::Matrix<float, 3, 1>;

	FV position;
	FV velocity;
	float radius;
	float mass;
	bool isFixed;
	int id;
	int cellHashId;
	float snowPortion;
	int neighborMax;
	bool hasBrokenBond;
	float d;  // durability variable

	Particle() 
		: position(FV(0.f, 0.f, 0.f))
		, velocity(FV(0.f, 0.f, 0.f))
		, radius(0.05f)
		, mass(0.0125f)  //0.785
		, isFixed(false)
		, id(1)
		, cellHashId(-1)
		, snowPortion(1.f)
		, neighborMax(-1)
		, hasBrokenBond(false)
		, d(1.f)
	{}
};