Real-time Particle-based Snow Simulation with Vulkan
====================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture**



**Qiaosen Chen:** [LinkedIn](https://www.linkedin.com/in/qiaosen-chen-725699141/)

**Haoyu Sui:** [LinkedIn](http://linkedin.com/in/haoyu-sui-721284192)

<!-- Tested on: Windows 10, i5-9600K @ 3.70GHz 16GB, RTX 2070 SUPER 8GB   -->
----

![](presentations/imgs/corner1.gif)

**Brief description**

The goal of this project is to achieve a real-time particle-based snow simulator. The simulation is expected to be accelerated by using GPU for attributes computation. Vulkan is used to visualize the snow particles in the simulation process.

Our project has following goals:

- Achieve a real-time, particle-based method to simulate snow dynamics on the GPU.
- Use Vulkan compute shader to accelerate the simulation and visualize snow particles in real time with Vulkan as well.
- Compare the performance of using CPU, CUDA and Vulkan compute shader for attributes computation.

**Motivation:**

Based on our interest in simulation, we chose this topic, and this is also a good opportunity for us to explore a new graphics API, Vulkan. 

We have done some physical simulation projects before, such as water and colloidal objects, but if you use the CPU for calculation, it often takes a long time to simulate a few seconds of movement, and only offline rendering can be achieved. Using CUDA can speed up and achieve real-time effects, but if visualization is performed at the same time, such as using OpenGL or Vulkan, because it involves frequent data copying, reading and writing, it will also greatly affect efficiency. So we consider using Vulkan's compute pipeline to accelerate the calculation, and use Vulkan for visualization, hoping to improve performance.



**Implementation overview**

The simulation is duvided into steps seen in the following algorithm:
	
	while animating do
		MovepParticles();
		NeighborSearch();
		CohesionAndRepulsion();
		HandleCollision();
		Compression();
	end

**Building the Project**

- Clone the repo 

- Open the CMake GUI to configure the project

- Make the "Source" directory points to the directory
Click Configure.

- Select your Visual Studio version, and x64 for your platform. 
Click Generate.

- If generation was successful, there should now be a Visual Studio solution (.sln) file in the build directory that you just created. Open this with Visual Studio.

- Build. 

**Third parties**

- Vulkan
- GLM
- Eigen
- [tiny-obj](https://github.com/tinyobjloader/tinyobjloader)

----

**References**

[Real-time particle-based snow simulation on the GPU](https://www.diva-portal.org/smash/get/diva2:1320769/FULLTEXT01.pdf)

[A material point method for snow simulation](https://www.math.ucla.edu/~jteran/papers/SSCTS13.pdf)

[Nvidia: use GPU to simulate fluid](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)

[Vulkan Tutorial](https://vulkan-tutorial.com/Introduction)
