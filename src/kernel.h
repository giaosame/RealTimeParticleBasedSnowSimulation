#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>
#include "Particle.h"
#include "vertex.h"
#include <device_functions.h>


namespace Boids {
    void initSimulation(int N);
    void stepSimulationNaive(float dt);
    void stepSimulationScatteredGrid(float dt);
    void stepSimulationCoherentGrid(float dt);
    void copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities);
    void copyParticlesToVBO(float* vbodptr_positions, float* vbodptr_velocities);

    void advanceOneStep(float dt);
    void copyParticlesToDevice(const std::vector<Particle>& particles);
    void copyParticlesToHost(Vertex* verts, const int numVerts);

    void endSimulation();
    void unitTest();
}
