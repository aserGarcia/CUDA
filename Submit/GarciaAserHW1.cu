#include <stdio.h>

const char* boolToTxt(int b){
    if (b)
				return "Enabled";
		return "Disabled";
}

int main(void){
		cudaDeviceProp prop;

		int count;
		cudaGetDeviceCount(&count);
		printf("\nNumber of Devices: %d\n", count);
		for(int i=0; i<count; i++){
				cudaGetDeviceProperties(&prop, i);
				printf("\n ---Device %d Information---\n", i);
				printf("Name: %s\n", prop.name);				
				
				//Thread/Blocks
        printf("Shared Mem Per Block: %lu\n", prop.sharedMemPerBlock);
				printf("Registers Per Block: %d\n", prop.regsPerBlock);
				printf("Registers Per MultiProcessor: %d\n", prop.regsPerMultiprocessor);
				printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
				printf("Max Threads per MultiProcessor: %d\n\n", prop.maxThreadsPerMultiProcessor);
				
				printf("Warp Size: %d\n", prop.warpSize);
				printf( "Max thread dimensions:(%d, %d, %d)\n",
					prop.maxThreadsDim[0], prop.maxThreadsDim[1],
					prop.maxThreadsDim[2] );
				printf( "Max grid dimensions:(%d, %d, %d)\n\n",
					prop.maxGridSize[0], prop.maxGridSize[1],
					prop.maxGridSize[2] );
					
				//Memory
				printf("Total Global Mem: %lu\n", prop.totalGlobalMem);
				printf("Total Constant Memory: %lu\n", prop.totalConstMem);
				printf("Total Managed Memory: %d\n", prop.managedMemory);
				printf("Shared Memory Per Block: %lu\n", prop.sharedMemPerBlock);
				printf("Shared Memory Per MultiProcessor: %lu\n", prop.sharedMemPerMultiprocessor);
				printf("Device can Map Host Memory: %s\n", boolToTxt(prop.canMapHostMemory));
				printf("Error Correcting code Mem: %s\n", boolToTxt(prop.ECCEnabled));
				printf("Memory Bus Width: %d\n", prop.memoryBusWidth);
				printf("Memory Pitch: %lu\n\n", prop.memPitch);
				
				//Computational Info
				printf("Major Compute Capability: %d\n", prop.major);
				printf("Minor Compute Capability: %d\n", prop.minor);
				printf("ClockRate: %d\n", prop.clockRate);
				printf("MultiProcessor Count: %d\n", prop.multiProcessorCount);
				printf("Device Overlap: %d\n", prop.deviceOverlap);

				printf("Kernel Execution Timeout: %s\n", boolToTxt(prop.kernelExecTimeoutEnabled));
				printf("Concurrent Kernels: %d\n", prop.concurrentKernels);
			  				
	  }
	  return 0;
}

