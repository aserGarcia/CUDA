//nvcc GarciaAserHW3.cu -o temp -lglut -lGL -lm

#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//window size
unsigned int window_width = 1024;
unsigned int window_height = 1024;


__device__ float color(float x, float y, float2 seed){
	float maxMag, t1;
	float maxCount = 600.0; //escape threshold
	float mag = 0.0;
	float count = 0.0;
	maxMag = 10.0;

	while(mag < maxMag && count < maxCount){
		t1 = x;
//---color---z_new = z_old^2 + seed
		x = x*x - y*y + seed.x;
		y = (2.0*t1*y) + seed.y;
		mag = sqrtf(x*x+y*y);
		count++;
	}
	return (count/maxCount)*(mag/maxMag); 
}

__global__ void kernel(float *pixels, float2 seed){
	float scale = 1.5;
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x+y*gridDim.x;
	
//---kernel---taking (x,y) -> complexSpace
	float jx = (float)(scale*(1.0-2.0*x/gridDim.x));
	float jy = (float)(scale*(1.0-2.0*y/gridDim.y));

	float c = color(jx, jy, seed);
//---kernel---storing into pixel col array
	pixels[offset*3+0] = 0.0; //Red
	pixels[offset*3+1] = c*(170.0/255.0); //Green
	pixels[offset*3+2] = c; //Blue
}

void display(void){
	float *pixels_GPU, *pixels_CPU;
	float x,y;
	const unsigned int mem_size = 3*window_width*window_height*sizeof(float);
	float2 seed_display;
	dim3 dimGrid(window_width,window_height);

//---display---allocating memory on dev&host
	pixels_CPU = (float*)malloc(mem_size);
	cudaMalloc(&pixels_GPU, mem_size);

//---display---forloop iterates over larger mandlebrot boundary
	for(float t = 0; t<6; t+=0.01){
//---display---resetting memory
		memset(pixels_CPU, 0.0, mem_size);
		cudaMemset(pixels_GPU, 0.0, mem_size);

//---display---parametric cycloid boundary eq for mandlebrot set
		x = 0.5*cos(t)-(0.25)*cos(2*t);
		y = 0.5*sin(t)-(0.25)*sin(2*t);
		seed_display = make_float2(x,y);

		kernel<<<dimGrid, 1>>>(pixels_GPU, seed_display);
		cudaMemcpy(pixels_CPU, pixels_GPU, mem_size, cudaMemcpyDeviceToHost);
//---display---drawing windows
		glDrawPixels(window_width, window_height, GL_RGB, GL_FLOAT, pixels_CPU);
		glFlush();
	}
	cudaFree(pixels_GPU);
	free(pixels_CPU);
}

int main(int argc, char** argv){
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(window_width, window_height);
   	glutCreateWindow("Fractal walk");
	glutDisplayFunc(display);
	glutMainLoop();  
	return 0;
}

