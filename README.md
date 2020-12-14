Real-time Particle-based Snow Simulation with Vulkan
====================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture**



**Qiaosen Chen:** [LinkedIn](https://www.linkedin.com/in/qiaosen-chen-725699141/)

**Haoyu Sui:** [LinkedIn](http://linkedin.com/in/haoyu-sui-721284192)

**All tests are performed:** Windows 10, i5-9600K @ 3.70GHz 16GB, RTX 2070 SUPER 8GB 

----

![](presentations/imgs/double_1.gif)

In the above scene, there are a total of 7,792 particles. For rendering, we load a spherical model for each particle, so there are a total of 7,792 * 180 = 140,2560 vertices. The fps is about 115. If we just render the particles as points, it could be much faster.

| Heart |  Ball | Box |
|---|---| ---|
|![](presentations/imgs/heart_1.gif)|![](presentations/imgs/writeball_1.gif)| ![](presentations/imgs/writecube_1.gif)|

### Brief Description

The project achieves a real-time particle-based snow simulator. The simulation is accelerated by using Vulkan compute shader for attributes computation, and the visulaization is implemented with Vulkan as well.

Our project has following goals:

- Achieve a real-time, particle-based method to simulate snow dynamics with Vulkan.
- Use Vulkan compute shader to accelerate the simulation and visualize snow particles in real time with Vulkan as well.
- Compare the performance of using CPU, CUDA and Vulkan compute shader for attributes computation.

### Motivation:

Based on our interest in simulation, we chose this topic, and this is also a good opportunity for us to explore a new graphics API, Vulkan. 

We have done some physical simulation projects before, such as water and colloidal objects simulation, but if we use the CPU for calculation, it often takes a long time to simulate a few seconds of movement, and only offline rendering can be achieved. Using CUDA can speed up and achieve real-time effects, but if visualization is performed at the same time, such as using OpenGL or Vulkan, because it involves frequent data copying, reading and writing, it will also greatly affect efficiency. So we consider using Vulkan's compute pipeline to accelerate the calculation, and use Vulkan for visualization, hoping to improve performance.



### Implementation Overview  

The simulation processes are duvided into the following steps:
	
	while animating do
		Find neighbors;
		Evaluate cohesive forces;
		Evaluate compressive forces;
		Handle collisions;
		Move particles;
	end



Vulkan compute pipelines are used to accelerate the attributes compuation, they are structured as follows:

	resetCellVertexCompute.comp
		- Reset the elements of cellVertArray and cellVertCount buffers to 0

	fillCellVertexInfo.comp
		- Find the cell id of particles and fill the cellVertArray buffer with particle ids

	physicsCompute.comp
		- Compute cohesive forces, compressive forces, update radius, move particles

	sphereVertexCompute.comp
		- Update the position of the vertices sphere 


### Results  

In this project, we achieved three different implementations of this simulation processes:

-  off-line computation with CPU + Houdini.

-  calculation and visualization based on CUDA + OpenGL.

-  calculation and visualization based on Vulkan graphics pipelines + Vulkan compute pipelines.

Here we will show our results and analyze their efficiency and performance. 

**Results of different implementations:**

| Vulkan compute and graphics pipeline |  CPU + Houdini | CUDA + OpenGL (rendering points) |
|---|---| ---|
|![](presentations/imgs/corner_sphere.gif)|![](presentations/imgs/softSnow_houdini.gif)| ![](presentations/imgs/corner_points.gif)|

**Results of different particle number:**

| 1,000 Particles |  8,000 Particles | 27,000 Particles |
|---|---| ---|
|![](presentations/imgs/sphere_10.gif)|![](presentations/imgs/sphere_20.gif)| ![](presentations/imgs/corner_sphere.gif)|

| 64,000 Particles |  125,000 Particles | 
|---|---| 
|![](presentations/imgs/sphere_40.gif)|![](presentations/imgs/sphere_50.gif)|


**Performance Analysis**

 |    | CPU (without rendering) | CUDA + OpenGL | Vulkan graphics + compute shaders |
| ------------- |-------------|-------------|-------------|
| Computing time per frame    | 29,600 ms    | 29.9 ms     | 2.3 ms     | 

All three implementations have 27,000 particles in the scene, and the CPU implementation only has computation part.

Compared with the use of GPU, the implementation method of CPU is very inefficient, and it is impossible to complete real-time functions. It can be seen that in simulation-related projects, reasonable use of GPU for calculations can significantly improve efficiency.

But when we need to do visualization, there are also significant differences in efficiency between different GPU acceleration and visualization API combinations. Because the combination of CUDA and OpenGL involves frequent and large amounts of data copy operations, the efficiency is relatively low. However, the Vulkan compute pipeline can be used to directly perform parallel attributes computation in the GPU, without the need to transfer the calculated data back to the CPU, and then the graphics pipeline can directly obtain the vertex data in the GPU, thus saving a lot of time and improving the efficiency.

![](presentations/imgs/particleNum_analysis.png)

The blue line represents the rendering of each particle into a sphere, which itself contains 180 vertices. The yellow line means rendering each particle into a point.

This picture shows the simulation of different particle numbers and the change in the time required to calculate each frame. As the number of particles increases, the time required increases. 



### Difficulties and Harvest 

One of the difficult parts of the project is to use computer shader to find neighbors of each particles. In our CUDA implementation We establish a model similar to the method introduced in [Particle Simulation using CUDA](http://developer.download.nvidia.com/assets/cuda/files/particles.pdf) for neighborhood computation wherein a virtual grid is established. A unique value is computed for each particle which maps it to a unique cell in the grid . This is quite straight-forward. But when we try to implement it in Vulkan compute pipeline, we encountered some problems, such as the key and value sorting problems, and the read-write control problems of the parallel sorting process.

So we changed a method, using GLSL's AtomicAdd operation, the purpose of atomic operation is to prevent multiple shader instances from accessing or reading the same memory area at the same time, causing access conflicts. Thus we have implemented an atomic counter to record the storage index of the next particle in each cell, so as to ensure that all recording operations will not affect each other.

Our main gain is the understanding and learning of Vulkan. Our project did not use some of the frame codes. We wrote it from the sketch, because we thought it would be more conducive for us to learn how to config and use Vulkan graphics and compute pipelines. Although it took us more time writing some basic framework code and debugging errors, these problems also let us learn a lot, such as using the validation layer to help to check the errors, and also found some useful tools that are helpful to debug the shader, such as RenderDoc. I think these will be of great help in our future study and work.


### Build the Project 

	1. Clone the repo 

	2. Open the CMake GUI to configure the project.

	3. Make the "Where is the source code" option points to the project folder.

	4. Create a "build" folder, and make the "Where to build binaries" option points to it.

	5. Click Configure.

	6. Select your Visual Studio version, and x64 for your platform. 

	7. Click Generate.

	8. If generation was successful, there should now be a Visual Studio solution (.sln) file in the build directory that you just created. Open this with Visual Studio.

	9. Build and run.

### Operation instruction 

- Change the rendering mode to render points or spheres. In main.cpp file:
	-  Make RENDER_USING_POINTS false: render spheres.
	-  Make RENDER_USING_POINTS true: render points.

- After running the code, the camera view can be changed by mouse operation:
	-  The left mouse button is used to rotate the scene.
	-  The right mouse button is used to zoom in and out.

### Third parties 

- Vulkan
- GLM
- Eigen
- [tiny-obj](https://github.com/tinyobjloader/tinyobjloader)

----

### References 

[Real-time particle-based snow simulation on the GPU](https://www.diva-portal.org/smash/get/diva2:1320769/FULLTEXT01.pdf)

[A material point method for snow simulation](https://www.math.ucla.edu/~jteran/papers/SSCTS13.pdf)

[Nvidia: use GPU to simulate fluid](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)

[Vulkan Tutorial](https://vulkan-tutorial.com/Introduction)

Many thanks and recommendations to the [RenderDoc](https://renderdoc.org/) graphics debugger. It quite useful for us to capture each frame and inspect details of application using Vulkan, OpenGL ES, etc.