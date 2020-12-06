#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <string>
#include <fstream>
#include "particle.h"

#include <map>
#include <algorithm>

#define PI 3.1415926f

class SnowSimSystem {
public:
	using FV = Eigen::Matrix<float, 3, 1>;

	std::vector<Particle> particles;

	std::map<int, std::vector<int>> cellHashValues;

	std::map<int, std::vector<int>> neighbors;
	

	// boundry of the space
	float width = 5.f;   // Y
	float length = 5.f;  // X
	float height = 5.f;  // Z
	FV startPoint = FV(0.f, 0.f, 0.f);

	float radius_snow = 0.05f;
	float radius_ice = 0.025f;
	float gridSize = 2 * radius_snow;

	float neighborRadius = 1.5f * gridSize;

	float youngs_modulus_snow = 25000.f;
	float youngs_modulus_ice = 35000.f;
	float cohesive_strength_snow = 625.f;
	float cohesive_strength_ice = 3750.f;

	float Kq = 0.00005f;
	float FminW = 0.12275f;
	float FmaxW = 10000.f;

	float damping = 0.95f;
	float boundry_damping = 0.5f;

	FV gravity = FV(0.f, -9.82f, 0.f);

	float Kf = 50.f;
	
	float angle_of_repose = 38.f / 180.f * PI;

	// For each particle, find its neighboors 
	// 1. Estabilish a virtual grid, for each particle
	// 2. Calculate a hash value based on its cell id
	// 3. Then sort the array by the hash value
	// 4. Find the start address of each cell id
	void findNeighboors() 
	{
		int cellCountX = int(ceil(length / gridSize));
		int cellCountY = int(ceil(width / gridSize));
		int cellCountZ = int(ceil(height / gridSize));

		cellHashValues.clear();
		neighbors.clear();

		// get particle ids in each cell
		for (int i = 0; i < particles.size(); ++i)
		{
			int indexX = (int)(particles[i].position(0) / gridSize);
			int indexY = (int)(particles[i].position(1) / gridSize);
			int indexZ = (int)(particles[i].position(2) / gridSize);

			int cellHashValue = indexZ * cellCountX * cellCountY + indexY * cellCountX + indexX;

			cellHashValues[cellHashValue].push_back(i);
		}
		// std::cout << cellHashValues[4210].size() << std::endl;

		// get neighbors of each particle
		for (int i = 0; i < particles.size(); ++i)
		{
			int indexX = (int)(particles[i].position(0) / gridSize);
			int indexY = (int)(particles[i].position(1) / gridSize);
			int indexZ = (int)(particles[i].position(2) / gridSize);
			//std::cout << " indexZ: " << indexZ << std::endl;

			//find neighbors by querying its current and 26 adjacent cells
			for (int x = -1; x <= 1; x++)
			{
				for (int y = -1; y <= 1; y++)
				{
					for (int z = -1; z <= 1; z++)
					{
						int curIdxX = indexX + x;
						int curIdxY = indexY + y;
						int curIdxZ = indexZ + z;
						int curCellId = curIdxZ * cellCountX * cellCountY + curIdxY * cellCountX + curIdxX;

						if (cellHashValues.count(curCellId) != 0)
						{
							std::vector<int> neighborTmp = cellHashValues[curCellId];
							for (int j = 0; j < neighborTmp.size(); j++)
							{
								int idTmp = neighborTmp[j];
								float dis = (particles[i].position - particles[idTmp].position).norm();
								if(dis <= neighborRadius && idTmp != i)
									neighbors[i].push_back(idTmp);
							}
						}
					}
				}
			}
			if (neighbors.count(i) != 0)
			{
				float nCurr = (float)(neighbors[i].size());
				float nMax = (float)(particles[i].neighborMax);
				if ((nCurr / nMax) < 0.75f && (nCurr / nMax) > 0)
					particles[i].hasBrokenBond = true;
				else
					particles[i].neighborMax = (int)(nCurr > nMax ? nCurr : nMax);

				//if (i == 0)
				//	std::cout <<  " nCurr: " << nCurr << std::endl;
			}
		}
	}

	void evaluateCohesiveForces(std::vector<FV>& cohesiveForces, std::vector<FV>& positiveForces, std::vector<FV>& negitiveForces) 
	{
		cohesiveForces.clear();
		cohesiveForces.resize(particles.size(), FV::Zero());
		positiveForces.clear();
		positiveForces.resize(particles.size(), FV::Zero());
		negitiveForces.clear();
		negitiveForces.resize(particles.size(), FV::Zero());

		for (int i = 0; i < particles.size(); ++i)
		{
			// for all neighbor particle k of i 
			if (neighbors.count(i) != 0)
			{
				std::vector<int> neighborVec = neighbors[i];
				for (int j = 0; j < neighborVec.size(); j++)
				{
					int neighborId = neighborVec[j];
					float dis = (particles[i].position - particles[neighborId].position).norm();

					if (particles[i].hasBrokenBond == false && particles[neighborId].hasBrokenBond == false && dis > (particles[i].radius + particles[neighborId].radius))
					{
						// calculate coheisve and tangential contant forces
						FV dir = (particles[i].position - particles[neighborId].position).normalized();
						float overlapDis = particles[i].radius + particles[neighborId].radius - dis;
						//FV forces = 1.f * youngs_modulus_snow * radius * overlapDis * dir;
						float Ei = youngs_modulus_snow * particles[i].snowPortion + youngs_modulus_ice * (1 - particles[i].snowPortion);
						float Ej = youngs_modulus_snow * particles[neighborId].snowPortion + youngs_modulus_ice * (1 - particles[neighborId].snowPortion);
						FV forces = (Ei * particles[i].radius + Ej * particles[neighborId].radius) / 2.f * overlapDis * dir;

						float cohesive_strength_i = cohesive_strength_snow * particles[i].snowPortion + cohesive_strength_ice * (1 - particles[i].snowPortion);
						float cohesive_strength_j = cohesive_strength_snow * particles[neighborId].snowPortion + cohesive_strength_ice * (1 - particles[neighborId].snowPortion);
						float condition1 = -1.f * (Ei * particles[i].radius + Ej * particles[neighborId].radius) / 2.f * overlapDis;
						float condition2 = 4.f * (cohesive_strength_i * particles[i].radius * particles[i].radius + cohesive_strength_j * particles[neighborId].radius * particles[neighborId].radius) / 2.f;
						if (condition1 < condition2)
							cohesiveForces[i] += forces;
						else
							cohesiveForces[i] += FV::Zero();
						continue;
					}
					
					// if i and its neighbor overlap
					else if (dis < (particles[i].radius + particles[neighborId].radius))
					{
						// calculate coheisve and tangential contant forces
						FV dir = (particles[i].position - particles[neighborId].position).normalized();
						float overlapDis = particles[i].radius + particles[neighborId].radius - dis;
						//FV forces = 1.f * youngs_modulus_snow * radius * overlapDis * dir;
						float Ei = youngs_modulus_snow * particles[i].snowPortion + youngs_modulus_ice * (1 - particles[i].snowPortion);
						float Ej = youngs_modulus_snow * particles[neighborId].snowPortion + youngs_modulus_ice * (1 - particles[neighborId].snowPortion);
						FV forces = (Ei * particles[i].radius + Ej * particles[neighborId].radius) / 2.f * overlapDis * dir;

						float cohesive_strength_i = cohesive_strength_snow * particles[i].snowPortion + cohesive_strength_ice * (1 - particles[i].snowPortion);
						float cohesive_strength_j = cohesive_strength_snow * particles[neighborId].snowPortion + cohesive_strength_ice * (1 - particles[neighborId].snowPortion);
						float condition1 = -1.f * (Ei * particles[i].radius + Ej * particles[neighborId].radius) / 2.f * overlapDis;
						float condition2 = 4.f * (cohesive_strength_i * particles[i].radius * particles[i].radius + cohesive_strength_j * particles[neighborId].radius * particles[neighborId].radius) / 2.f;
						if (condition1 < condition2)
							cohesiveForces[i] += forces;
						else
							cohesiveForces[i] += FV::Zero();

						// update compressive forces
						if (forces(0) > 0)
							positiveForces[i](0) += forces(0);
						else
							negitiveForces[i](0) += forces(0);

						if (forces(1) > 0)
							positiveForces[i](1) += forces(1);
						else
							negitiveForces[i](1) += forces(1);

						if (forces(2) > 0)
							positiveForces[i](2) += forces(2);
						else
							negitiveForces[i](2) += forces(2);

						//positiveForces[i](0) = forces(0) > 0 ? (positiveForces[i](0) + forces(0)) : positiveForces[i](0);

						// evaluate tensile contact force
						FV vi = particles[i].velocity;
						FV vj = particles[neighborId].velocity;
						if (vi != vj)
						{
							FV ut = -1.f * (vi - vj).normalized();

							float tanCoeff = tan(angle_of_repose);

							FV Ft = ut * forces.norm() * tanCoeff * 1.f;
							cohesiveForces[i] += Ft;
						}
					}
				}
			}
		}
	}

	void evaluateCompression(std::vector<float>& compressiveForces, std::vector<FV>& positiveForces, std::vector<FV>& negitiveForces)
	{
		compressiveForces.clear();
		compressiveForces.resize(particles.size(), 0);

		for(int i = 0; i < particles.size(); ++i)
		{
			float minXSquare = std::min(positiveForces[i](0) * positiveForces[i](0), negitiveForces[i](0) * negitiveForces[i](0));
			float minYSquare = std::min(positiveForces[i](1) * positiveForces[i](1), negitiveForces[i](1) * negitiveForces[i](1));
			float minZSquare = std::min(positiveForces[i](2) * positiveForces[i](2), negitiveForces[i](2) * negitiveForces[i](2));

			compressiveForces[i] = std::sqrt(minXSquare + minYSquare + minZSquare);

			float p = compressiveForces[i] / (3.14159f * particles[i].radius * particles[i].radius);

			float pi = 100.f * particles[i].snowPortion + 900.f * (1 - particles[i].snowPortion);
			float e = 2.71828183f;
			float Dpi = FminW + FmaxW * ((std::pow(e, (pi / 100.f - 1)) - 0.000335f) / 2980.96f);

			//float d = particles[i].d;
			if (compressiveForces[i] > Dpi)
			{
				particles[i].d -= Kq * p;
				particles[i].radius = particles[i].d * radius_snow + (1 - particles[i].d) * radius_ice;
				particles[i].snowPortion = particles[i].d;
			}
		}
	}

	void evaluateThermodynamics()
	{

	}

	void moveParticles(std::vector<FV>& cohesiveForces, float dt, std::vector<FV>& positiveForces, std::vector<FV>& negitiveForces)
	{
		for (int i = 0; i < particles.size(); ++i)
		{
			if (particles[i].isFixed)
			{
				particles[i].velocity = FV::Zero();
			}
			else
			{
				FV vNew = particles[i].velocity + (cohesiveForces[i] / particles[i].mass + gravity) * dt;
				// test
				//vNew(0) = 0.f;
				//vNew(2) = 0.f;
				//if (i == 10)
				//	std::cout << f_cohesive[i](1) << std::endl;
				FV xNew = particles[i].position + vNew * dt;

				// collision check

				// hit the ground
				if (xNew(1) < particles[i].radius || xNew(1) > length - particles[i].radius)
				{
					// moving the particle to the surface of the wall
					//xNew(1) = xNew(1) < particles[i].radius ? particles[i].radius : length - particles[i].radius;
					if (xNew(1) < particles[i].radius)
					{
						xNew(1) = particles[i].radius;
						positiveForces[i](1) = -1.f * negitiveForces[i](1);
					}
					else
					{
						xNew(1) = length - particles[i].radius;
						negitiveForces[i](1) = -1.f * positiveForces[i](1);
					}

					// change the velocity
					FV vn = FV::Zero();
					FV vt = vNew;
					vn(1) = -1.f * vNew(1);
					vt(1) = 0.f;

					vNew = vn + vt * std::max(0.f, 1.f - Kf * (vn.norm() / vt.norm()));
					vNew *= boundry_damping;
				}

				if (xNew(0) < particles[i].radius || xNew(0) > length - particles[i].radius)
				{
					// moving the particle to the surface of the wall
					//xNew(0) = xNew(0) < particles[i].radius ? particles[i].radius : length - particles[i].radius;
					if (xNew(0) < particles[i].radius)
					{
						xNew(0) = particles[i].radius;
						positiveForces[i](0) = -1.f * negitiveForces[i](0);
					}
					else
					{
						xNew(0) = length - particles[i].radius;
						negitiveForces[i](0) = -1.f * positiveForces[i](0);
					}

					// change the velocity
					FV vn = FV::Zero();
					FV vt = vNew;
					vn(0) = -1.f * vNew(0);
					vt(0) = 0.f;

					vNew = vn + vt * std::max(0.f, 1.f - Kf * (vn.norm() / vt.norm()));
					vNew *= boundry_damping;
				}

				if (xNew(2) < particles[i].radius || xNew(2) > length - particles[i].radius)
				{
					// moving the particle to the surface of the wall
					//xNew(2) = xNew(2) < particles[i].radius ? particles[i].radius : length - particles[i].radius;
					if (xNew(2) < particles[i].radius)
					{
						xNew(2) = particles[i].radius;
						positiveForces[i](2) = -1.f * negitiveForces[i](2);
					}
					else
					{
						xNew(2) = length - particles[i].radius;
						negitiveForces[i](2) = -1.f * positiveForces[i](2);
					}

					// change the velocity
					FV vn = FV::Zero();
					FV vt = vNew;
					vn(2) = -1.f * vNew(2);
					vt(2) = 0.f;

					vNew = vn + vt * std::max(0.f, 1.f - Kf * (vn.norm() / vt.norm()));
					vNew *= boundry_damping;
				}

				// update the velocity and position
				particles[i].velocity = vNew * damping;
				particles[i].position = xNew;

			}
		}
	}

	// write POLY file
	void dumpPoly(std::string filename) 
	{
		std::ofstream fs;
		fs.open(filename);
		fs << "POINTS\n";
		int count = 0;
		for (auto P : particles) {
			fs << ++count << ":";
			for (int i = 0; i < 3; i++) {
				fs << " " << P.position(i);
			}
			fs << "\n";
		}
		fs << "POLYS\n";
		count = 0;
		fs << "END\n";
		fs.close();
	}
};