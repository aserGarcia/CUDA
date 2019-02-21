//nvcc GarciaAserHW8.cu -o temp -lglut -lm -lGLU -lGL

/*------------------------------------
		TO DO
1. dev_r to float3 to increase speed
2. Global vars -> #define
3. Redo lines 118 to 131
------------------------------------*/

#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#define CHUNKSIZE   1024

int BLOCKSIZE = 265;
int WINDOW_SIZE = 1024;
int FULL_DATA_SIZE = WINDOW_SIZE*WINDOW_SIZE*3; //each pixel has three floats

/*-------------------------------------------
		KERNEL
-------------------------------------------*/
__global__ void kernel(float *a, float *b, float *r){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int offset = idx*3;
//Checking pixel difference between files
    if(idx < CHUNKSIZE){
        if (a[offset+0] != b[offset+0] || a[offset+1] != b[offset+1] || a[offset+2] != b[offset+2]){
            r[offset+0] = 1.0;
            r[offset+1] = 1.0;
            r[offset+2] = 1.0;
        }
        else{
            r[offset+0] = 0.0;
            r[offset+1] = 0.0;
            r[offset+2] = 0.0;
        }
    }
}

/*-------------------------------------------
	DISPLAY TO SCREEN
-------------------------------------------*/
void display()
{
//*********   READING FILE   ***********
    float *file1, *file2;
    FILE *bitmapFile;
    
    cudaHostAlloc(&file1, FULL_DATA_SIZE*sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc(&file2, FULL_DATA_SIZE*sizeof(float), cudaHostAllocDefault);

    bitmapFile = fopen("File1", "rb");
    fread(file1, sizeof(float), FULL_DATA_SIZE, bitmapFile);
    bitmapFile = fopen("File2", "rb");
    fread(file2, sizeof(float), FULL_DATA_SIZE, bitmapFile);

    fclose(bitmapFile);

//*********   KERNEL CALL   ***********
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    float *result;
    float *dev_a0, *dev_b0, *dev_r0; //GPU stream 0
    float *dev_a1, *dev_b1, *dev_r1; //GPU stream 1

    cudaMalloc(&dev_a0, CHUNKSIZE*3*sizeof(float));
    cudaMalloc(&dev_b0, CHUNKSIZE*3*sizeof(float));
    cudaMalloc(&dev_r0, CHUNKSIZE*3*sizeof(float));
    cudaMalloc(&dev_a1, CHUNKSIZE*3*sizeof(float));
    cudaMalloc(&dev_b1, CHUNKSIZE*3*sizeof(float));
    cudaMalloc(&dev_r1, CHUNKSIZE*3*sizeof(float));

    cudaHostAlloc(&result, FULL_DATA_SIZE*sizeof(float), cudaHostAllocDefault);

    for(long i =0; i<CHUNKSIZE*(CHUNKSIZE*3-2); i += CHUNKSIZE*2){
        //copying up to stream0
        cudaMemcpyAsync(dev_a0, file1+i,
                        CHUNKSIZE*3*sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream0);
        cudaMemcpyAsync(dev_a1, file1+i+CHUNKSIZE,
                        CHUNKSIZE*3*sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream1);

        //copying up to stream1
        cudaMemcpyAsync(dev_b0, file2+i,
                        CHUNKSIZE*3*sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream0);
        cudaMemcpyAsync(dev_b1, file2+i+CHUNKSIZE,
                        CHUNKSIZE*3*sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream1);

        kernel<<<CHUNKSIZE/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_r0);
        kernel<<<CHUNKSIZE/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_r1);

        cudaMemcpyAsync(result+i, dev_r0,
                        CHUNKSIZE*3*sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream0);
        cudaMemcpyAsync(result+i+CHUNKSIZE, dev_r1,
                        CHUNKSIZE*3*sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream1);
        
    }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);


    float tol = 0.001;
    float px;
    for(int i =0; i < CHUNKSIZE*3; i+=3){
        for(int j =0; j<CHUNKSIZE*3; j+=3){
            if(result[i+j*CHUNKSIZE]){
                px = file2[i+j*CHUNKSIZE];
                for(int j = 1; j<27; j++){
                    if (abs(px-1.0/float(j))<tol){
                        printf("%c", (char)(j+64));
                    }
                    
                }
                printf(" ");
            }
        }
    }

	glDrawPixels(WINDOW_SIZE, WINDOW_SIZE, GL_RGB, GL_FLOAT, result);
    glFlush();
    
    cudaFreeHost(file1);
    cudaFreeHost(file2);
    cudaFreeHost(result);
    cudaFree(dev_a0);
    cudaFree(dev_b0);
    cudaFree(dev_r0);
    cudaFree(dev_a1);
    cudaFree(dev_b1);
    cudaFree(dev_r1);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
}

int main(int argc, char** argv)
{
   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WINDOW_SIZE, WINDOW_SIZE);
   	glutCreateWindow("BitMap");
   	glutDisplayFunc(display);
   	glutMainLoop();
}
