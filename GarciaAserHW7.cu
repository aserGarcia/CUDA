//nvcc GarciaAserHW7.cu -o temp -lm; ./temp (runs)
/*
----If it is timing out (watchdog) follow steps below----
1. sudo sublime /etc/X11/xorg.conf
(if it does not exist run command: sudo nvidia-xconfig)

2. in the "Device" section add [Option "Interactive" "0"]  no brackets

3. save any file progress you have

4. restart device manager: sudo systemctl restart display-manager
*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>

#define MAX_BLOCKSIZE 64 //limited amount of registers per thread force a low block count.
#define PI 3.141592654
#define NO 0
#define YES 1	
		
//constants globals
int NUMBER_OF_RUNS = 1010;
const float CENTRAL_ATRACTION_FORCE  = 0.1;
const float REPULSIVE_SLOPE = 50000.0;
const float DIAMETER_PS = 1.0; // Diameter of polystyrene spheres 1 micron
const float DIAMETER_NIPAM = 0.08; // Diameter of polyNIPAM microgel particles 80 nanometers
const float START_RADIUS_OF_INVIRONMENT = 5.0;
const float START_SEPERATION_TOL = 1.1;
//device globals
__constant__ float g_drag = 15.25714286;
__constant__ float g_max_attraction = 376.5;
__constant__ float DT = 0.0001;
__constant__ float MASS = 1.0; //estimate with density 1.05g per cm cubed
__constant__ float MAX_INITIAL_VELOCITY = 1.0;
__constant__ int NUMBER_OF_BODIES = 6;

struct Body{
    float4 pos;
    float3 vel;
    float3 force;
    __device__ void setPos(float4 p){ pos = p; }
    __device__ void setVel(float3 v){ vel = v; }
    __device__ void setForce(float3 f){ force = f;}
    __device__ void updatePosVel(float3 f){
        vel = make_float3( vel.x + DT*(f.x - g_drag*vel.x)/pos.w, vel.y + DT*(f.y - g_drag*vel.y)/pos.w, vel.z + DT*(f.z - g_drag*vel.z)/pos.w );
        pos = make_float4(pos.x + DT*vel.x, pos.y + DT*vel.y, pos.z + DT*vel.z, pos.w);
    }
};

__device__ void get_forces(Body *bodies){
	int i,j;
	float dx,dy,dz,r,total_force,d;
	
	for(i = 0; i < NUMBER_OF_BODIES - 1; i++){
		for(j = i + 1; j < NUMBER_OF_BODIES; j++){
			dx = bodies[j].pos.x - bodies[i].pos.x;
			dy = bodies[j].pos.y - bodies[i].pos.y;
			dz = bodies[j].pos.z - bodies[i].pos.z;
		
			r = sqrt(dx*dx + dy*dy + dz*dz);
			
			if(r < DIAMETER_PS){
				total_force =  REPULSIVE_SLOPE*r - REPULSIVE_SLOPE*DIAMETER_PS + g_max_attraction;
			}
			else if (r < DIAMETER_PS + DIAMETER_NIPAM){
				total_force =  -(g_max_attraction/DIAMETER_NIPAM)*r + (g_max_attraction/DIAMETER_NIPAM)*(DIAMETER_PS + DIAMETER_NIPAM);
			}
			else total_force = 0.0;

            bodies[i].setForce(make_float3(bodies[i].force.x + total_force*dx/r, bodies[i].force.y + total_force*dy/r, bodies[i].force.z + total_force*dz/r));
            bodies[j].setForce(make_float3(bodies[j].force.x - total_force*dx/r, bodies[j].force.y - total_force*dy/r, bodies[j].force.z -= total_force*dz/r));
		}
	}
	
	for(i = 0; i < NUMBER_OF_BODIES; i++){
		d = sqrt(bodies[i].pos.x*bodies[i].pos.x + bodies[i].pos.y*bodies[i].pos.y + bodies[i].pos.z*bodies[i].pos.z);
		bodies[i].force.x += -CENTRAL_ATRACTION_FORCE*g_max_attraction*bodies[i].pos.x/d;
		bodies[i].force.y += -CENTRAL_ATRACTION_FORCE*g_max_attraction*bodies[i].pos.y/d;
		bodies[i].force.z += -CENTRAL_ATRACTION_FORCE*g_max_attraction*bodies[i].pos.z/d;

		bodies[i].updatePosVel(bodies[i].force);
    }
}

__global__ void set_initial_conditions(int *dev_bins, curandState *d_state){
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    float r0, r1, r2;
	int i,j,ok_config;
    float angle1, angle2, rad, dx, dy, dz;
    float t = 0.0, dt = 0.0001, tStop = 25.0;
    curand_init(1234, tid, 0, &d_state[tid]);

    Body bodies[6];
	
	ok_config = NO;
	while(ok_config == NO){
		for(i = 0; i < NUMBER_OF_BODIES; i++){
            r0 = (float)curand_uniform(&d_state[tid]);
            r1 = (float)curand_uniform(&d_state[tid]);
            r2 = (float)curand_uniform(&d_state[tid]);

			rad = START_RADIUS_OF_INVIRONMENT*r0;
			angle1 = PI*r1;
            angle2 = 2.0*PI*r2;
            bodies[i].setPos(make_float4(rad*sinf(angle1)*cosf(angle2),rad*sinf(angle1)*sinf(angle2),rad*cosf(angle1), MASS));
		}
			
		ok_config = YES;
		for(i = 0; i < NUMBER_OF_BODIES - 1; i++){
			for(j = i + 1; j < NUMBER_OF_BODIES; j++){
				dx = bodies[j].pos.x-bodies[i].pos.x;
				dy = bodies[j].pos.y-bodies[i].pos.y;
				dz = bodies[j].pos.z-bodies[i].pos.z;
				if(sqrt(dx*dx + dy*dy + dz*dz) <= START_SEPERATION_TOL) ok_config = NO;
			}
		}
	}

	for(i = 0; i < NUMBER_OF_BODIES; i++){
        r0 = curand_uniform(&d_state[tid]);
        r1 = curand_uniform(&d_state[tid]);
        r2 = curand_uniform(&d_state[tid]);

		rad = MAX_INITIAL_VELOCITY*r0;
		angle1 = PI*r1;
        angle2 = 2.0*PI*r2;
        bodies[i].setVel(make_float3(rad*sinf(angle1)*cosf(angle2), rad*sinf(angle1)*sinf(angle2), rad*cosf(angle1)));
    }	

    while(t < tStop){
		for(i = 0; i < NUMBER_OF_BODIES; i++)
			bodies[i].setForce(make_float3(0.0,0.0,0.0));
	
		get_forces(bodies);			
		t += dt;
	}

	float total_body_to_body_distance = 0.0;
	for(i = 0; i < NUMBER_OF_BODIES - 1; i++){
		for(j = i + 1; j < NUMBER_OF_BODIES; j++){
			dx = bodies[j].pos.x-bodies[i].pos.x;
			dy = bodies[j].pos.y-bodies[i].pos.y;
			dz = bodies[j].pos.z-bodies[i].pos.z;
			
			total_body_to_body_distance += sqrt(dx*dx + dy*dy + dz*dz);
		}
	}
   
    //putting it into histogram
    if(total_body_to_body_distance<16.7) atomicAdd(&dev_bins[1], 1);
    else if(total_body_to_body_distance>16.9) atomicAdd(&dev_bins[2], 1);
    else atomicAdd(&dev_bins[0], 1);
}

void nbody(){   
    curandState *d_state;
    cudaMalloc(&d_state, sizeof(curandState));

    int bins[3];
    int *dev_bins;

    int dimBlock = NUMBER_OF_RUNS<MAX_BLOCKSIZE? NUMBER_OF_RUNS: MAX_BLOCKSIZE;
    int dimGrid = NUMBER_OF_RUNS<MAX_BLOCKSIZE? 1:(NUMBER_OF_RUNS-1)/MAX_BLOCKSIZE + 1;

    cudaMalloc(&dev_bins, 3*sizeof(int));
    cudaMemset(dev_bins, 0, 3*sizeof(int));

    set_initial_conditions<<<dimGrid, dimBlock>>>(dev_bins, d_state);
    cudaMemcpy(bins, dev_bins, 3*sizeof(int), cudaMemcpyDeviceToHost);

    printf("---------Histogram---------\n");
    printf("Poly-tetrahedrons: %d\nOctahedrons:       %d\nOther:             %d\n", bins[0], bins[1], bins[2]);
    printf("DONE");

    cudaFree(dev_bins);
}

int main(int argc, char** argv){
    float g_max_attraction = 376.5;
	printf("\nThe Zero force distance is %f\n",(REPULSIVE_SLOPE*DIAMETER_PS - g_max_attraction)/REPULSIVE_SLOPE);
	nbody();
	return 0;
}
