#include <iostream>

//The __global__ makes it available for 
//both CPU and GPU
__global__ void kernel(void) {
}

int main(void) {
  kernel<<<1,1,>>>();
  printf("Hello, World!\n");
  return 0;
}
