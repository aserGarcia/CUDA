#Self Assembly of Polysterene Spheres
This program outputs the histogram values for shape self assembly of polysterene spheres. Each thread runs a simulation and determines the shape after assembly. Three shapes may occur:
1. Polytetrahedron
2. Octohedron
3. Unbounded
The positions and velocities are initialized randomly for each simulation using cuda's random number generator from a uniform distribution.
#What I learned
* Particle Simulation
* Polysterene self assembly
* Cuda structures with device functions
* Curand generator
* Watchdog timer (kernel exec timeout safety feature)
