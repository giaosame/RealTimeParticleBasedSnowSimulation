Real-time Particle-based Snow Simulation with Vulkan (still working on it)
====================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture**



**Qiaosen Chen:** [LinkedIn](https://www.linkedin.com/in/qiaosen-chen-725699141/)

**Haoyu Sui:** [LinkedIn](http://linkedin.com/in/haoyu-sui-721284192)

<!-- Tested on: Windows 10, i5-9600K @ 3.70GHz 16GB, RTX 2070 SUPER 8GB   -->
----

**Brief description**

The goal of this project is to achieve a real-time particle-based snow simulator. The simulation is expected to be accelerated by using GPU for attributes computation. Vulkan is used to visualize the snow particles in the simulation process.

![](presentations/imgs/softSnow.gif)

**Implementation overview**

The simulation is duvided into steps seen in the following algorithm:
	
	while animating do
		MovepParticles();
		NeighborSearch();
		CohesionAndRepulsion();
		HandleCollision();
		Thermodynamics();
		Compression();
	end

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
