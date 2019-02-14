/*
nvcc -lineinfo <name>.cu -o temp //for cuda-memcheck
cuda-memcheck ./temp |more	//gives you more detailed errors
*/
#include <stdio.h>

#define N 1025

//---------Kernel Function-----------
__global__ void add(float *a, float* b, float* c, int n){
	//getting specific thread number
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	
	//making sure we are in range of vector length
	if (id < n){
		c[id] = a[id] + b[id];
	}
}

//---------Main Function-------------------
int main(void){
//----main-----Find max/min threads on device(s)
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

//----main-----Thread and block numbers
	int maxThreads = prop.maxThreadsPerBlock;
	int blockNr = ((N-1)/maxThreads)+1;
	dim3 dimGrid(blockNr);	
	dim3 dimBlock(maxThreads); 

//----main-----Declaring variables-----------------
	float *A_CPU, *B_CPU, *C_CPU; //Host pointers
	float *A_GPU, *B_GPU, *C_GPU; //Device pointers
	
//----main-----Allocoating memory for pointers-------
	A_CPU = (float*)malloc(N*sizeof(float));//Host ptrs
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	cudaMalloc(&A_GPU, N*sizeof(float));//Device ptrs
	cudaMalloc(&C_GPU, N*sizeof(float));
	cudaMalloc(&B_GPU, N*sizeof(float));
	
//----main-----Getting the work done---------
	//loading values into vectors
	for(unsigned int i=0; i<N; i++){
		A_CPU[i] = (float)i;
		B_CPU[i] = (float)i*i;
	}
	
	//copying vectors to GPU
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	
	//Executing kernel function
	add<<<dimGrid,dimBlock>>>(A_GPU, B_GPU, C_GPU, N);
	
	//recieving added vector values of GPU
	cudaMemcpy(C_CPU, C_GPU, N*sizeof(float), cudaMemcpyDeviceToHost);
	
	//summing values of added vectors
	float sum = 0.0;
	for(int i = 0; i<N; i++){
		sum += C_CPU[i];
	}
	
	printf("The sum of the two vectors is: %lf", sum);
//----main-----Freeing allocated memory
	free(A_CPU);
	free(B_CPU);
	free(C_CPU);
	cudaFree(A_GPU);
	cudaFree(B_GPU);
	cudaFree(C_GPU);
	
	return 0;
}
