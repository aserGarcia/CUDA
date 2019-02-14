// To compile: nvcc HW4.cu -o temp; ./temp
#include <sys/time.h>
#include <stdio.h>

//---global vars---access for GPU
const int N = 2000;

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

const int threadsPerBlock = prop.maxThreadsPerBlock;
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

__global__ void dotProd(float *A_GPU, float *B_GPU, float *C_GPU){
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
			sh_mem[sh_mem_id] += sh_mem[sh_mem_id + i];
			__syncthreads();
		}
		i /= 2;
	}

	//---dotProd---inserts block sum into C_GPU to add later
	if(sh_mem_id == 0)
		C_GPU[blockIdx.x] = sh_mem[0];
}

int main()
{
	long id;
	float *A_CPU, *B_CPU, *C_CPU, sumErrytin; //Pointers for memory on the Host
	// Your variables start here.
	float *A_GPU, *B_GPU, *C_GPU;
	// Your variables stop here.
	
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	for(id = 0; id < N; id++) {A_CPU[id] = 1; B_CPU[id] = 3;}
	
	// Your code starts here.

	//---main---mallocGPU
	cudaMalloc(&A_GPU, N*sizeof(float));
	CUDAErrorCheck("cudaMalloc A_GPU");
	cudaMalloc(&B_GPU, N*sizeof(float));
	CUDAErrorCheck("cudaMalloc B_GPU");
	cudaMalloc(&C_GPU, blocksPerGrid*sizeof(float));
	CUDAErrorCheck("cudaMalloc C_GPU");

	//---main---memCpy host->dev
	cudaMemcpy(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	CUDAErrorCheck("A_CPU --> A_GPU cpy");
	cudaMemcpy(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	CUDAErrorCheck("B_CPU --> B_GPU cpy");
	
	//---main---kernel exec
	dotProd<<<blocksPerGrid, threadsPerBlock>>>(A_GPU, B_GPU, C_GPU);
	CUDAErrorCheck("dotProd kernel exec");

	//---main---memCpy dev->host
	cudaMemcpy(C_CPU, C_GPU, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost);
	CUDAErrorCheck("C_GPU --> C_CPU cpy");

	//---main---sumErrytin
	sumErrytin = 0.0;
	for(int i=0; i<blocksPerGrid; i++){
		sumErrytin += C_CPU[i];
	}

	printf("value: %f", sumErrytin);

	//---main---free mem gpu
	cudaFree(A_GPU);
	CUDAErrorCheck("freeing A_GPU");
	cudaFree(B_GPU);
	CUDAErrorCheck("freeing B_GPU");
	cudaFree(C_GPU);
	CUDAErrorCheck("freeing C_GPU");

	//---main---free mem cpu
	free(A_CPU);
	CUDAErrorCheck("freeing A_CPU");
	free(B_CPU);
	CUDAErrorCheck("freeing B_CPU");
	free(C_CPU);
	CUDAErrorCheck("freeing C_CPU");
	// Your code stops here.
	
	return(0);
}
