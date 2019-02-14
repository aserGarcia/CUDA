
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INF 2e10f
#define SPHERES 20
#define rnd(x) (x*rand()/RAND_MAX)

const int DIM = 1024;

struct Sphere{
    float r,b,g;
    float radius;
    float x,y,z;
};

//Sphere *s;
__constant__ Sphere s_const[SPHERES];

//method checks if pixel hits a sphere
__device__ float hit(float px, float py, float x, float y, float z, float r, float *n){
    float dx = px - x;
    float dy = py - y;
    if(dx*dx + dy*dy < r*r){
        float dz = sqrtf(r*r - dx*dx - dy*dy);
       *n = dz / r;
        return dz+z;
    }
    return -INF;
}

//kernel uses global memory
__global__ void kernel(float *pixels, Sphere *s){
    int idx = blockIdx.x;
    int idy = blockIdx.y;
    int offset = 3*(idx+idy*gridDim.x);

    float px = idx - DIM/2;
    float py = idy - DIM/2;

    float r=0, b=0, g=0;
    float maxz = -INF;
    for(int i =0; i < SPHERES; i++){
        float n;
        float d = hit(px, py, s[i].x, s[i].y, s[i].z, s[i].radius, &n);
        if(d > maxz){
            r = s[i].r * n;
            g = s[i].g * n;
            b = s[i].b * n;
        }
    }
    pixels[offset+0] = r;
    pixels[offset+1] = g;
    pixels[offset+2] = b;
}

//kernel uses constant memory
__global__ void kernelConst(float *pixels){
    int idx = blockIdx.x;
    int idy = blockIdx.y;
    int offset = 3*(idx+idy*gridDim.x);

    //creates a coordinate system on the image
    float px = idx - DIM/2;
    float py = idy - DIM/2;

    float r=0, b=0, g=0;
    float maxz = -INF;
    for(int i =0; i < SPHERES; i++){
        float n;
        float d = hit(px, py, s_const[i].x, s_const[i].y, s_const[i].z, s_const[i].radius, &n);
        if(d > maxz){
            r = s_const[i].r * n;
            g = s_const[i].g * n;
            b = s_const[i].b * n;
        }
    }
    pixels[offset+0] = r;
    pixels[offset+1] = g;
    pixels[offset+2] = b;
}


void display(void){
//variables for drawing
    float *pixels_GPU, *pixels_CPU;
    unsigned pix_memsize = 3*DIM*DIM*sizeof(float);
    unsigned sphere_memsize = SPHERES*sizeof(Sphere);
    Sphere *s_cpu, *s;

//variables for timing
    cudaEvent_t start, start2, stop, stop2;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
//allocating GPU
    cudaMalloc(&pixels_GPU, pix_memsize);
    cudaMalloc(&s, sphere_memsize);

//allocating CPU
    cudaMallocHost(&pixels_CPU, pix_memsize);
    cudaMallocHost(&s_cpu, sphere_memsize);

//initalizing sphere characteristics
    for(int i = 0; i < SPHERES; i++){
        s_cpu[i].r = rnd(1.0f);
        s_cpu[i].g = rnd(1.0f);
        s_cpu[i].b = rnd(1.0f);
        
        s_cpu[i].x = rnd(1000.0f)-500;
        s_cpu[i].y = rnd(1000.0f)-500;
        s_cpu[i].z = rnd(1000.0f)-500;
        s_cpu[i].radius = rnd(100.0f)+20;
    }

//copying from CPU to GPU
    cudaMemcpy(s, s_cpu, sphere_memsize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(s_const, s_cpu, sphere_memsize);

    dim3 dimGrid(DIM,DIM);

//global memory kernel time
    cudaEventRecord(start, 0);

    kernel<<<dimGrid, 1>>>(pixels_GPU, s);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time for global mem: %fms\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

//constant memory global time
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);

    kernelConst<<<dimGrid, 1>>>(pixels_GPU);

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);

    float elapsedTime2;
    cudaEventElapsedTime(&elapsedTime2, start2, stop2);
    printf("Elapsed time for constant mem: %fms\n", elapsedTime2);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

//drawing to screen
    cudaMemcpy(pixels_CPU, pixels_GPU, pix_memsize, cudaMemcpyDeviceToHost);

    glDrawPixels(DIM, DIM, GL_RGB, GL_FLOAT, pixels_CPU);
    glFlush();
    
    cudaFree(pixels_GPU);
    cudaFree(pixels_CPU);
    cudaFree(s);
    cudaFree(s_cpu);
while(1);
}

int main(int argc, char** argv){
    glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(DIM, DIM);
   	glutCreateWindow("spheres");
	glutDisplayFunc(display);
	glutMainLoop();  
	return 0;
}
