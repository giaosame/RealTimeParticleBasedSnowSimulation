#pragma once
#include <Eigen/Sparse>
#include <unsupported/Eigen/IterativeSolvers>

#include <sys/stat.h>
#include <iostream>
#include "SnowSimSystem.h"
#include <functional>
#include <direct.h>

class SimulationDriver {
public:
	using FV = Eigen::Matrix<float, 3, 1>;

	std::string test;
	SnowSimSystem ss;

	float dt;

	SimulationDriver()
	: dt(0.0017f)
	{}

	void run()
	{
		float accumulate_t = 0;
		_mkdir("output/");
		std::string output_folder = "output/" + test;
		_mkdir(output_folder.c_str());
		std::string filename = output_folder + "/" + std::to_string(0) + ".poly";
		ss.dumpPoly(filename);
		for (int frame = 1; frame < 200; frame++)
		{
			std::cout << "Frame " << frame << std::endl;
			int N_substeps = (int)((1.f / 24.f) / dt);
			for (int step = 1; step <= N_substeps; step++)
			{
				//std::cout << "Step " << step << std::endl;
				advanceOneStep();
				accumulate_t += dt;
				//return;
			}
			_mkdir("output/");
			std::string output_folder = "output/" + test;
			_mkdir(output_folder.c_str());
			std::string filename = output_folder + "/" + std::to_string(frame) + ".poly";
			ss.dumpPoly(filename);
			std::cout << std::endl;
		}
	}

	void advanceOneStep()
	{
		std::vector<FV> f_cohesive;
		std::vector<FV> f_positive;
		std::vector<FV> f_negitive;
		std::vector<float> f_compressive;

		// find neighboors
		ss.findNeighboors();

		// compute cohesion forces, thermodynamics, compression
		ss.evaluateCohesiveForces(f_cohesive, f_positive, f_negitive);
		
		// update velocity and position
		ss.moveParticles(f_cohesive, dt, f_positive, f_negitive);

		// compute compression forces and update radius
		ss.evaluateCompression(f_compressive, f_positive, f_negitive);
	}
};