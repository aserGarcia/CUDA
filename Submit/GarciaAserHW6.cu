// To compile: nvcc HW4.cu -o temp; ./temp
/*32-bit floating point atomicAdd() supported on compute capability 2.x and higher*/
#include <sys/time.h>
#include <stdio.h>

//---global vars---access for GPU
const int N = 2000001;

const int threadsPerBlock = 1024;
const int blocksPerGrid = ((N-1)/threadsPerBlock)+1;


//error check func for methods
void CUDAErrorCheck(const char *message)
{
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    printf("\n CUDA ERROR in: %s -> %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

__global__ void reduce(float *A_GPU, float *result){
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if(id < gridDim.x){
		atomicAdd(result, A_GPU[id]);
	}
}

__global__ void dotProd(float *A_GPU, float *B_GPU){
	//---dotProd---will give the thread ability to share mem on block
	__shared__ float sh_mem[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	//---dotProd---limits at the nr_threads/Block
	int sh_mem_id = threadIdx.x;

	//---dotProd---store the product of id temporarily
	float temp = 0.0;
	while(tid < N){
		temp += A_GPU[tid] * B_GPU[tid];
		tid += blockDim.x * gridDim.x; //takes us to the id in the next block
	}
	__syncthreads();

	//---dotProd---set mem val for that id
	sh_mem[sh_mem_id] = temp;

	__syncthreads();

	//---dotProd---for any number vector
	//change
	int i = blockDim.x/2;
	while(i != 0){
		//will only execute if threadId within vector length
		if(sh_mem_id < i){
            //halfing the vector and adding the matching locations
			atomicAdd(&sh_mem[sh_mem_id], sh_mem[sh_mem_id+i]);
		}
		i /= 2;
		__syncthreads();
	}
	
	
	if (sh_mem_id==0){
		A_GPU[blockIdx.x] = sh_mem[sh_mem_id];
	}
	__syncthreads();
}

int main()
{
	long id;
	float *A_CPU, *B_CPU, r=0; //Pointers for memory on the Host
	// Your variables start here.
	float *A_GPU, *B_GPU, *r_gpu;
	// Your variables stop here.
	
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	for(id = 0; id < N; id++) {A_CPU[id] = 1; B_CPU[id] = 2;}
	
	// Your code starts here.

	//---main---mallocGPU
	cudaMalloc(&A_GPU, N*sizeof(float));
	CUDAErrorCheck("cudaMalloc A_GPU");
	cudaMalloc(&B_GPU, N*sizeof(float));
	CUDAErrorCheck("cudaMalloc B_GPU");
	cudaMalloc(&r_gpu, sizeof(float));
	CUDAErrorCheck("cudaMalloc r_gpu");

	//---main---memCpy host->dev
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	CUDAErrorCheck("A_CPU --> A_GPU cpy");
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	CUDAErrorCheck("B_CPU --> B_GPU cpy");
	cudaMemcpy(r_gpu, &r, sizeof(float), cudaMemcpyHostToDevice);
	CUDAErrorCheck("r --> r_gpu");
	
	//---main---kernel exec
	dotProd<<<blocksPerGrid, threadsPerBlock>>>(A_GPU, B_GPU);
	CUDAErrorCheck("dotProd kernel exec");
	reduce<<<blocksPerGrid, threadsPerBlock>>>(A_GPU, r_gpu);
	CUDAErrorCheck("reduce kernel exec");

	//---main---memCpy dev->host
	cudaMemcpy(&r, r_gpu, sizeof(float), cudaMemcpyDeviceToHost);
	CUDAErrorCheck("r_gpu --> r cpy");

	printf("value: %f", r);

	//---main---free mem gpu
	cudaFree(A_GPU);
	CUDAErrorCheck("freeing A_GPU");
	cudaFree(B_GPU);
	CUDAErrorCheck("freeing B_GPU");
	cudaFree(r_gpu);
	CUDAErrorCheck("freeing r_gpu");

	//---main---free mem cpu
	free(A_CPU);
	CUDAErrorCheck("freeing A_CPU");
	free(B_CPU);
	CUDAErrorCheck("freeing B_CPU");
	// Your code stops here.
	
	return(0);
}