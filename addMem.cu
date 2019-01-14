 #include <iostream>
 #include "book.h"
 
 __global__ void add(int a, int b, int *c){
  *c = a+b;
 }
 
 int main(void){
  //integer to store kernel result
  int c;
  
  //device pointer to integer
  int *dev_c;
  
  //allocating memory on the device, args: pointer to pointer wanted, size of alloc
  cudaMalloc((void**)&dev_c, sizeof(int));
  
  //kernel call 1thrd, 1blk + args
  add<<<1,1>>>(2,7,dev_c);
  
  //copying memory from device to cpu args: pointer, pointer, size of copy, direction
  cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf( "2+7 = %d\n", c);
  
  //freeing memory previously allocated
  cudaFree(dev_c);
  
  return 0;
  
 }
 
