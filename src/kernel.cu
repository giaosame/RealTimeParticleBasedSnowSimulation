#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"
#include "Particle.h"
#include "vertex.h"
#include <device_functions.h>


// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

float cellSize = 0.1f;
float girdSideLength = 5.f;
float neighborRadius = 1.5f * cellSize;  // 0.15 = 3 * particle radius
float gridSideCount_temp = girdSideLength / cellSize; // 50
glm::vec3 gridMinimum_temp = glm::vec3(0.f); 
float gridInverseCellWidth_temp = 1.f / cellSize; // 10

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_shuffledPos;
glm::vec3* dev_shuffledVel1;

Particle* dev_particles;
Particle* dev_newParticles;
Vertex* dev_verts;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}


/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  int gridCellCount_temp = 125000;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount_temp * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount_temp * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  // dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaMalloc((void**)&dev_shuffledPos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_shuffledPos failed!");

  cudaMalloc((void**)&dev_shuffledVel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_shuffledVel1 failed!");

  cudaMalloc((void**)&dev_particles, N * sizeof(Particle));
  checkCUDAErrorWithLine("cudaMalloc dev_particles failed!");

  cudaMalloc((void**)&dev_newParticles, N * sizeof(Particle));
  checkCUDAErrorWithLine("cudaMalloc dev_newParticles failed!");

  cudaMalloc((void**)&dev_verts, N * sizeof(Vertex));
  checkCUDAErrorWithLine("cudaMalloc dev_verts failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyParticlesToVBO(int N, Particle* particles, float *vbo_p, float *vbo_v, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo_p[4 * index + 0] = particles[index].position.x + 0.3f;
    vbo_p[4 * index + 1] = particles[index].position.y + 0.3f;
    vbo_p[4 * index + 2] = particles[index].position.z + 0.3f;
    vbo_p[4 * index + 3] = 1.0f;

    vbo_v[4 * index + 0] = particles[index].velocity.x * s_scale;
    vbo_v[4 * index + 1] = particles[index].velocity.y * s_scale;
    vbo_v[4 * index + 2] = particles[index].velocity.z * s_scale;
    vbo_v[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}

void Boids::copyParticlesToVBO(float *vbodptr_positions, float *vbodptr_velocities)
{
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyParticlesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_particles, vbodptr_positions, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    int neighborCount = 0;
    glm::vec3 center = {0.f, 0.f, 0.f};
    glm::vec3 separate = { 0.f, 0.f, 0.f };
    glm::vec3 cohesion = { 0.f, 0.f, 0.f };

    glm::vec3 newVel = vel[iSelf];

    for (int i = 0; i < N; i++) 
    {
        if (i == iSelf) continue;
        float distance = glm::length(pos[i] - pos[iSelf]);

        if (distance < rule1Distance)
        {
            // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
            center += pos[i];
            neighborCount++;

            // Rule 2: boids try to stay a distance d away from each other
            if (distance < rule2Distance)
            {
                separate -= (pos[i] - pos[iSelf]);
            }

            // Rule 3: boids try to match the speed of surrounding boids
            cohesion += vel[i];
        }
    }

    if (neighborCount > 0)
    {
        center /= neighborCount;
        newVel += (center - pos[iSelf]) * rule1Scale;
        newVel += cohesion * rule3Scale;
    }

    newVel += separate * rule2Scale;
  
    return newVel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // Compute a new velocity based on pos and vel1
    glm::vec3 newVel = computeVelocityChange(N, index, pos, vel1) + vel1[index];

    // Clamp the speed
    float magnitude = glm::length(newVel);
    if (magnitude > maxSpeed) {
        newVel *= maxSpeed / magnitude;
    }

    // Record the new velocity into vel2. Question: why NOT vel1?
    vel2[index] = newVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    glm::vec3 posRel = pos[index] - gridMin;
    int xRel = std::floor(posRel.x * inverseCellWidth);
    int yRel = std::floor(posRel.y * inverseCellWidth);
    int zRel = std::floor(posRel.z * inverseCellWidth);

    gridIndices[index] = gridIndex3Dto1D(xRel, yRel, zRel, gridResolution);
    indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
     
    // int prevCell = -1, curCell = -1;

    // if (index == 0) {
    //     prevCell = -1;
    //     curCell = particleGridIndices[0];
    //     gridCellStartIndices[curCell] = index;
    // }
    // else
    // {
    //     prevCell = particleGridIndices[index - 1];
    //     curCell = particleGridIndices[index];
    //     if (prevCell != curCell)
    //     {
    //         gridCellStartIndices[curCell] = index;
    //         gridCellEndIndices[prevCell] = index - 1;
    //     }
    // }
    if (index == 0)
    {
        gridCellStartIndices[particleGridIndices[index]] = index;
    }
    else if(index == N - 1)
    {
        gridCellEndIndices[particleGridIndices[index]] = index;
    }
    else
    {
        if (particleGridIndices[index - 1] != particleGridIndices[index])
        {
            gridCellEndIndices[particleGridIndices[index - 1]] = index - 1;
            gridCellStartIndices[particleGridIndices[index]] = index;
        }
    }
    // __syncthreads();

}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    int neighborCountRule1 = 0, neighborCountRule3 = 0;
    glm::vec3 center = { 0.f, 0.f, 0.f };
    glm::vec3 separate = { 0.f, 0.f, 0.f };
    glm::vec3 cohesion = { 0.f, 0.f, 0.f };

    // - Identify the grid cell that this particle is in
    glm::vec3 gridPos = (pos[index] - gridMin) * inverseCellWidth;
    glm::vec3 posRel = glm::floor(gridPos);
    int cellIndex = gridIndex3Dto1D(posRel.x, posRel.y, posRel.z, gridResolution);

    int xDir = (gridPos - posRel).x > 0.5 ? 1 : 0,
        yDir = (gridPos - posRel).y > 0.5 ? 1 : 0,
        zDir = (gridPos - posRel).z > 0.5 ? 1 : 0;

    // - Identify which cells may contain neighbors. This isn't always 8.
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int z = -1; z <= 1; z++)
            {
                int X = posRel.x + x,
                    Y = posRel.y + y,
                    Z = posRel.z + z,
                    curCellIndex = cellIndex + x + y * gridResolution + z * gridResolution * gridResolution;

                curCellIndex += (xDir + yDir * gridResolution + zDir * gridResolution * gridResolution);
                X += xDir;
                Y += yDir;
                Z += zDir;

                if (X < 0 || X >= gridResolution
                    || Y < 0 || Y >= gridResolution
                    || Z < 0 || Z >= gridResolution)
                    continue;

                // - For each cell, read the start/end indices in the boid pointer array.
                int cellStart = gridCellStartIndices[curCellIndex];
                int cellEnd = gridCellEndIndices[curCellIndex];

                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance
                for (int c = cellStart; c <= cellEnd; c++)
                {
                    int i = particleArrayIndices[c];
                    if (i == index) continue;
                    float distance = glm::length(pos[i] - pos[index]);

                    // rule1
                    if (distance < rule1Distance)
                    {
                        center += pos[i];
                        neighborCountRule1++;
                    }

                    // rule2
                    if (distance < rule2Distance)
                    {
                        separate -= (pos[i] - pos[index]);
                    }

                    // rule3
                    if (distance < rule3Distance)
                    {
                        cohesion += vel1[i];
                        neighborCountRule3++;
                    }
                }
            }
        }

    }

    if (neighborCountRule1 > 0)
    {
        center = (center / (float)neighborCountRule1 - pos[index]) * rule1Scale;
    }

    separate  = separate * rule2Scale;

    if (neighborCountRule3 > 0)
    {
        cohesion = (cohesion / (float)neighborCountRule3) * rule3Scale;
    }

    glm::vec3 newVel = vel1[index] + center + separate + cohesion;

    // - Clamp the speed change before putting the new speed in vel2
    float magnitude = glm::length(newVel);
    if (magnitude > maxSpeed) {
        newVel *= maxSpeed / magnitude;
    }
    vel2[index] = newVel;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.

    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    int neighborCountRule1 = 0, neighborCountRule3 = 0;
    glm::vec3 center = { 0.f, 0.f, 0.f };
    glm::vec3 separate = { 0.f, 0.f, 0.f };
    glm::vec3 cohesion = { 0.f, 0.f, 0.f };

    // - Identify the grid cell that this particle is in
    glm::vec3 gridPos = (pos[index] - gridMin) * inverseCellWidth;
    glm::vec3 posRel = glm::floor(gridPos);
    int cellIndex = gridIndex3Dto1D(posRel.x, posRel.y, posRel.z, gridResolution);

    int xDir = (gridPos - posRel).x > 0.5 ? 1 : 0,
        yDir = (gridPos - posRel).y > 0.5 ? 1 : 0,
        zDir = (gridPos - posRel).z > 0.5 ? 1 : 0;

    // - Identify which cells may contain neighbors. This isn't always 8.
    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int z = -1; z <= 1; z++)
            {
                int X = posRel.x + x,
                    Y = posRel.y + y,
                    Z = posRel.z + z,
                    curCellIndex = cellIndex + x + y * gridResolution + z * gridResolution * gridResolution;

                curCellIndex += (xDir + yDir * gridResolution + zDir * gridResolution * gridResolution);
                X += xDir;
                Y += yDir;
                Z += zDir;

                if (X < 0 || X >= gridResolution
                    || Y < 0 || Y >= gridResolution
                    || Z < 0 || Z >= gridResolution)
                    continue;

                // - For each cell, read the start/end indices in the boid pointer array.
                int cellStart = gridCellStartIndices[curCellIndex];
                int cellEnd = gridCellEndIndices[curCellIndex];

                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance
                for (int c = cellStart; c <= cellEnd; c++)
                {
                    if (c == index) continue;
                    float distance = glm::length(pos[c] - pos[index]);

                    // rule1
                    if (distance < rule1Distance)
                    {
                        center += pos[c];
                        neighborCountRule1++;
                    }

                    // rule2
                    if (distance < rule2Distance)
                    {
                        separate -= (pos[c] - pos[index]);
                    }

                    // rule3
                    if (distance < rule3Distance)
                    {
                        cohesion += vel1[c];
                        neighborCountRule3++;
                    }
                }
            }
        }

    }

    if (neighborCountRule1 > 0)
    {
        center = (center / (float)neighborCountRule1 - pos[index]) * rule1Scale;
    }

    separate = separate * rule2Scale;

    if (neighborCountRule3 > 0)
    {
        cohesion = (cohesion / (float)neighborCountRule3) * rule3Scale;
    }

    glm::vec3 newVel = vel1[index] + center + separate + cohesion;

    // - Clamp the speed change before putting the new speed in vel2
    float magnitude = glm::length(newVel);
    if (magnitude > maxSpeed) {
        newVel *= maxSpeed / magnitude;
    }
    vel2[index] = newVel;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernUpdateVelocityBruteForce <<<fullBlocksPerGrid, blockSize>>> (numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos <<<fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel1);
  // TODO-1.2 ping-pong the velocity buffers
    std::swap(dev_vel1, dev_vel2);
}

// __global__ void kernComputeIndices(int N, int gridResolution,
//   glm::vec3 gridMin, float inverseCellWidth,
//   glm::vec3 *pos, int *indices, int *gridIndices) {
//     // TODO-2.1
//     // - Label each boid with the index of its grid cell.
//     // - Set up a parallel array of integer indices as pointers to the actual
//     //   boid data in pos and vel1/vel2
//     int index = threadIdx.x + (blockIdx.x * blockDim.x);
//     if (index >= N) {
//         return;
//     }

//     glm::vec3 posRel = pos[index] - gridMin;
//     int xRel = std::floor(posRel.x * inverseCellWidth);
//     int yRel = std::floor(posRel.y * inverseCellWidth);
//     int zRel = std::floor(posRel.z * inverseCellWidth);

//     gridIndices[index] = gridIndex3Dto1D(xRel, yRel, zRel, gridResolution);
//     indices[index] = index;
// }

__global__ void kernUpdateParticles(
  int N, 
  int gridResolution, 
  glm::vec3 gridMin,
  float inverseCellWidth, 
  float cellWidth,
  float neighborRadius,
  float dt,
  int *gridCellStartIndices,
  int *gridCellEndIndices,
  int *particleArrayIndices,
  int *particleGridIndices,
  Particle *particles,
  Particle *newParticles) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    Particle p = particles[index];
    // - Identify the grid cell that this particle is in
    glm::vec3 gridPos = (p.position - gridMin) * inverseCellWidth;
    glm::vec3 posRel = glm::floor(gridPos);
    int cellIndex = gridIndex3Dto1D(posRel.x, posRel.y, posRel.z, gridResolution);

    float youngs_modulus_snow = 25000.f;
    float youngs_modulus_ice = 35000.f;
    float cohesive_strength_snow = 625.f;
    float cohesive_strength_ice = 3750.f;
    float FminW = 0.12275f;
    float FmaxW = 10000.f;
    float radius_snow = 0.05f;
    float radius_ice = 0.025f;
    glm::vec3 gravity = glm::vec3(0.f, -20.82f, 0.f);
    float damping = 0.98f;
    float boundry_damping = 0.5f;
    float Kq = 0.00005f;
    float angle_of_repose = 38.f / 180.f * 3.1415926f;
    float Kf = 50.f;

    // __syncthreads();

    // array for neighbor
    const int neighborsNum = 100;
    int neighbors[neighborsNum] = {0};  // -1
    for(int i = 0 ; i < neighborsNum; ++i)
    {
      neighbors[i] = -1;
    }
    int neighborCount = 0;

    for (int x = -1; x <= 1; x++)
    {
        for (int y = -1; y <= 1; y++)
        {
            for (int z = -1; z <= 1; z++)
            {
                int X = posRel.x + x,
                    Y = posRel.y + y,
                    Z = posRel.z + z,
                    curCellIndex = cellIndex + x + y * gridResolution + z * gridResolution * gridResolution;

                if (X < 0 || X >= gridResolution
                    || Y < 0 || Y >= gridResolution
                    || Z < 0 || Z >= gridResolution)
                    continue;

                // - For each cell, read the start/end indices in the boid pointer array.
                // int cellStart = particleGridIndices[curCellIndex];
                // int cellEnd = particleGridIndices[curCellIndex];
                int cellStart = gridCellStartIndices[curCellIndex];
                int cellEnd = gridCellEndIndices[curCellIndex];

                if(cellStart == -1 || cellEnd == -1)
                  continue;

                for (int c = cellStart; c <= cellEnd; c++)
                {
                    int i = particleArrayIndices[c];

                    if (i == index) continue;
                    float distance = glm::length(particles[i].position - p.position);

                    if (distance < neighborRadius && neighborCount < neighborsNum)
                    {
                        neighbors[neighborCount] = i;
                        neighborCount++;
                    }
                }
            }
        }
    }
    // if(index == 0)
    // {
    //     printf("neighborCount : %d \n", neighborCount);
    // }
  

    float temp = (float)(neighborCount) / (float)(p.neighborMax);
    if (temp < 0.75f && temp > 0.f)
    {
        newParticles[index].hasBrokenBond = true;
    }
    else
    {
        newParticles[index].neighborMax = neighborCount > p.neighborMax ? neighborCount : p.neighborMax;
    }

    // compute cohesive forces
    glm::vec3 cohesiveForces = glm::vec3(0.f);
    glm::vec3 positiveForces = glm::vec3(0.f);
    glm::vec3 negitiveForces = glm::vec3(0.f);

    //printf("neighborCount: %d \n", neighborCount);
    if (neighborCount > 0)
    //if (false)
    { 
        for (int i = 0; i < neighborCount; ++i)
        {
            int neighborIdx = neighbors[i];
            const Particle& pNeighbor = particles[neighborIdx];

            float dist = glm::length(p.position - pNeighbor.position);

            if(p.hasBrokenBond == false && pNeighbor.hasBrokenBond == false && 
              dist > (p.radius + pNeighbor.radius))
            {
                // calculate coheisve and tangential contant forces
                glm::vec3 dir = glm::normalize(p.position - pNeighbor.position);
                float overlapDist = p.radius + pNeighbor.radius - dist;

                float Ei = youngs_modulus_snow * p.snowPortion + youngs_modulus_ice * (1 - p.snowPortion);
                float Ej = youngs_modulus_snow * pNeighbor.snowPortion + youngs_modulus_ice * (1 - pNeighbor.snowPortion);

                glm::vec3 forces = (Ei * p.radius + Ej * pNeighbor.radius) / 2.f * overlapDist * dir;

                float cohesive_strength_i = cohesive_strength_snow * p.snowPortion + cohesive_strength_ice * (1 - p.snowPortion);
                float cohesive_strength_j = cohesive_strength_snow * pNeighbor.snowPortion + cohesive_strength_ice * (1 - pNeighbor.snowPortion);

                float condition1 = -1.f * (Ei * p.radius + Ej * pNeighbor.radius) / 2.f * overlapDist;
                float condition2 = 4.f * (cohesive_strength_i * p.radius * p.radius + cohesive_strength_j * pNeighbor.radius * pNeighbor.radius) / 2.f;

                if (condition1 < condition2)
                  cohesiveForces += forces;
                else
                  cohesiveForces += glm::vec3(0.f);
            }
            else if(dist < (p.radius + pNeighbor.radius))
            {
                // calculate coheisve and tangential contant forces
                glm::vec3 dir = glm::normalize(p.position - pNeighbor.position);
                float overlapDist = p.radius + pNeighbor.radius - dist;

                float Ei = youngs_modulus_snow * p.snowPortion + youngs_modulus_ice * (1 - p.snowPortion);
                float Ej = youngs_modulus_snow * pNeighbor.snowPortion + youngs_modulus_ice * (1 - pNeighbor.snowPortion);

                glm::vec3 forces = (Ei * p.radius + Ej * pNeighbor.radius) / 2.f * overlapDist * dir;

                float cohesive_strength_i = cohesive_strength_snow * p.snowPortion + cohesive_strength_ice * (1 - p.snowPortion);
                float cohesive_strength_j = cohesive_strength_snow * pNeighbor.snowPortion + cohesive_strength_ice * (1 - pNeighbor.snowPortion);

                float condition1 = -1.f * (Ei * p.radius + Ej * pNeighbor.radius) / 2.f * overlapDist;
                float condition2 = 4.f * (cohesive_strength_i * p.radius * p.radius + cohesive_strength_j * pNeighbor.radius * pNeighbor.radius) / 2.f;

                if (condition1 < condition2)
                  cohesiveForces += forces;
                else
                  cohesiveForces += glm::vec3(0.f);

                // update compressive forces
                if (forces.x > 0)
                  positiveForces.x += forces.x;
                else
                  negitiveForces.x += forces.x;

                if (forces.y > 0)
                  positiveForces.y += forces.y;
                else
                  negitiveForces.y += forces.y;

                if (forces.z > 0)
                  positiveForces.z  += forces.z;
                else
                  negitiveForces.z  += forces.z;

                // evaluate tensile contact force
                glm::vec3 vi = p.velocity;
                glm::vec3 vj = pNeighbor.velocity;
                if (vi != vj)
                {
                  glm::vec3 ut = -1.f * glm::normalize(vi - vj);

                  float tanCoeff = glm::tan(angle_of_repose);

                  glm::vec3 Ft = ut * glm::length(forces) * tanCoeff * 1.f;
                  cohesiveForces += Ft;

                  if (Ft.x > 0)
                    positiveForces.x += Ft.x;
                  else
                    negitiveForces.x += Ft.x;

                  if (Ft.y > 0)
                    positiveForces.y += Ft.y;
                  else
                    negitiveForces.y += Ft.y;

                  if (Ft.z > 0)
                    positiveForces.z  += Ft.z;
                  else
                    negitiveForces.z  += Ft.z;
                }
            }
        }
    }

    

    // move particles
    if (p.isFixed)
    {
        newParticles[index].velocity = glm::vec3(0.f);
    }
    else
    {
        glm::vec3 vNew = p.velocity + (cohesiveForces / p.mass + gravity) * dt;
        glm::vec3 xNew = p.position + vNew * dt;

        // hit the ground
        if(xNew.y < p.radius || xNew.y > (5.f - p.radius))
        {
            if(xNew.y < p.radius)
            {
                xNew.y = p.radius;
                positiveForces.y = negitiveForces.y * -1.f;
            }
            else
            {
                xNew.y = 5.f - p.radius;
                negitiveForces.y = -1.f * positiveForces.y;
            }
            glm::vec3 vn = glm::vec3(0.f);
            glm::vec3 vt = vNew;

            vn.y = -1.f * vNew.y;
            vt.y = 0.f;

            vNew = vn + vt * glm::max(0.f, 1.f - Kf * (glm::length(vn) / glm::length(vt)));
            vNew *= boundry_damping;
        }

        if(xNew.x < p.radius || xNew.x > (5.f - p.radius))
        {
            if(xNew.x < p.radius)
            {
                xNew.x = p.radius;
                positiveForces.x = negitiveForces.x * -1.f;
            }
            else
            {
                xNew.x = 5.f - p.radius;
                negitiveForces.x = -1.f * positiveForces.x;
            }
            glm::vec3 vn = glm::vec3(0.f);
            glm::vec3 vt = vNew;

            vn.x = -1.f * vNew.x;
            vt.x = 0.f;

            vNew = vn + vt * glm::max(0.f, 1.f - Kf * (glm::length(vn) / glm::length(vt)));
            vNew *= boundry_damping;
        }

        if(xNew.z < p.radius || xNew.z > (5.f - p.radius))
        {
            if(xNew.z < p.radius)
            {
                xNew.z = p.radius;
                positiveForces.z = negitiveForces.z * -1.f;
            }
            else
            {
                xNew.z = 5.f - p.radius;
                negitiveForces.z = -1.f * positiveForces.z;
            }
            glm::vec3 vn = glm::vec3(0.f);
            glm::vec3 vt = vNew;

            vn.z = -1.f * vNew.z;
            vt.z = 0.f;

            vNew = vn + vt * glm::max(0.f, 1.f - Kf * (glm::length(vn) / glm::length(vt)));
            vNew *= boundry_damping;
        }

        //xNew = p.position;
        newParticles[index].velocity = vNew * damping;
        newParticles[index].position = xNew;
    }

    // compression 
    float minXSquare = glm::min(positiveForces.x * positiveForces.x, negitiveForces.x * negitiveForces.x);
    float minYSquare = glm::min(positiveForces.y * positiveForces.y, negitiveForces.y * negitiveForces.y);
    float minZSquare = glm::min(positiveForces.z * positiveForces.z, negitiveForces.z * negitiveForces.z);

    float compressiveForces = glm::sqrt(minXSquare + minYSquare + minZSquare);

    float p_temp = compressiveForces / (3.1415926f * p.radius * p.radius);

    float pi = 100.f * p.snowPortion + 900.f * (1 - p.snowPortion);
    float e = 2.71828183f;
    float Dpi = FminW + FmaxW * ((glm::pow(e, (pi / 100.f - 1)) - 0.000335f) / 2980.96f);

    // update radius
    if (compressiveForces > Dpi)
      {
          newParticles[index].d = p.d - Kq * p_temp;
          newParticles[index].radius = p.d * radius_snow + (1 - p.d) * radius_ice;
          newParticles[index].snowPortion = p.d;
      }
}

__global__ void kernComputeParticlesIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  Particle* particles, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    glm::vec3 posRel = particles[index].position - gridMin;
    int xRel = (int)(std::floor(posRel.x * inverseCellWidth));
    int yRel = (int)(std::floor(posRel.y * inverseCellWidth));
    int zRel = (int)(std::floor(posRel.z * inverseCellWidth));

    gridIndices[index] = gridIndex3Dto1D(xRel, yRel, zRel, gridResolution);
    indices[index] = index;
}

__global__ void kernCopyParticlesToVertices(int N, Particle* particles, Vertex* verts)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // verts[index].pos = particles[index].position;
}


void Boids::advanceOneStep(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 gridCount((gridCellCount + blockSize - 1) / blockSize);

  // find neighbors
  
  float cellSize = 0.1f;
  float girdSideLength = 5.f;
  float neighborRadius = 1.5f * cellSize;  // 0.15 = 3 * particle radius
  float gridSideCount_temp = girdSideLength / cellSize; // 50
  glm::vec3 gridMinimum_temp = glm::vec3(0.f); 
  float gridInverseCellWidth_temp = 1.f / cellSize; // 10

  // label each particle with its grid index
  kernComputeParticlesIndices <<<fullBlocksPerGrid, blockSize>>> (numObjects, (int)(gridSideCount_temp), gridMinimum_temp,
    gridInverseCellWidth_temp, dev_particles, dev_particleArrayIndices, dev_particleGridIndices);

  // cudaDeviceSynchronize();

  // Unstable key sort using Thrust
  thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
  thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);

  // cudaDeviceSynchronize();

  kernResetIntBuffer <<<gridCount, blockSize >>> (numObjects, dev_gridCellStartIndices, -1);
  kernResetIntBuffer <<<gridCount, blockSize >>> (numObjects, dev_gridCellEndIndices, -1);

  // cudaDeviceSynchronize();

  // find the start and end indices of each cell's data
  kernIdentifyCellStartEnd <<<fullBlocksPerGrid, blockSize >>> (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);

  // cudaDeviceSynchronize();

  kernUpdateParticles <<<fullBlocksPerGrid, blockSize>>> (
    numObjects,
    gridSideCount_temp,
    gridMinimum_temp,
    gridInverseCellWidth_temp,
    cellSize,
    neighborRadius,
    dt,
    dev_gridCellStartIndices, 
    dev_gridCellEndIndices, 
    dev_particleArrayIndices,
    dev_particleGridIndices,
    dev_particles,
    dev_newParticles
    );

  std::swap(dev_particles, dev_newParticles);
  kernCopyParticlesToVertices << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particles, dev_verts);
}


void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 gridCount((gridCellCount + blockSize - 1) / blockSize);

    // - label each particle with its array index as well as its grid index.
    //   Use 2x width grids.
    kernComputeIndices <<<fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
    thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);

    kernResetIntBuffer <<<gridCount, blockSize >>> (numObjects, dev_gridCellStartIndices, -1);
    kernResetIntBuffer <<<gridCount, blockSize >>> (numObjects, dev_gridCellEndIndices, -1);

    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernIdentifyCellStartEnd <<<fullBlocksPerGrid, blockSize >>> (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered <<<fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices,
        dev_pos, dev_vel1, dev_vel2);
    // - Update positions
    kernUpdatePos <<<fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel2);
    // - Ping-pong buffers as needed 
    std::swap(dev_vel1, dev_vel2);
}

__global__ void kernShufflePosVel(int N, int* indices,
    glm::vec3* pos, glm::vec3* vel1,
    glm::vec3* shuffledPos, glm::vec3* shuffledVel1)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    int particleIndex = indices[index];
    shuffledPos[index] = pos[particleIndex];
    shuffledVel1[index] = vel1[particleIndex];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:

    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 gridCount((gridCellCount + blockSize - 1) / blockSize);
    kernComputeIndices <<<fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

    // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
    //   are welcome to do a performance comparison.
    thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
    thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);

    // - Naively unroll the loop for finding the start and end indices of each
    //   cell's data pointers in the array of boid indices
    kernResetIntBuffer <<<gridCount, blockSize >>> (numObjects, dev_gridCellStartIndices, -1);
    kernResetIntBuffer <<<gridCount, blockSize >>> (numObjects, dev_gridCellEndIndices, -1);

    // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
    //   the particle data in the simulation array.
    //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
    kernIdentifyCellStartEnd <<<fullBlocksPerGrid, blockSize >>> (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    kernShufflePosVel <<<fullBlocksPerGrid, blockSize >>> (numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_shuffledPos, dev_shuffledVel1);

    cudaMemcpy(dev_pos, dev_shuffledPos, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_vel1, dev_shuffledVel1, numObjects * sizeof(glm::vec3), cudaMemcpyDeviceToDevice);

    // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_pos, dev_vel1, dev_vel2);

    // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

    // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
    std::swap(dev_vel1, dev_vel2);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_particles);
  cudaFree(dev_newParticles);
  cudaFree(dev_verts);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}

void Boids::copyParticlesToDevice(const std::vector<Particle>& particles)
{
    cudaMemcpy(dev_particles, particles.data(), sizeof(Particle) * particles.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_newParticles, particles.data(), sizeof(Particle) * particles.size(), cudaMemcpyHostToDevice);
    checkCUDAErrorWithLine("cudaMemcpy particles failed!");
}

void Boids::copyParticlesToHost(Vertex* verts, const int numVerts)
{
    cudaMemcpy(verts, dev_verts, sizeof(Vertex) * numVerts, cudaMemcpyDeviceToHost);
    checkCUDAErrorWithLine("cudaMemcpy particles to host failed!");
}