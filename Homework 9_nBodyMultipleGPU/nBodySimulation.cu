// Optimized using shared memory and on chip memory
// nvcc nBodySimulation.cu -o nBody -lglut -lm -lGLU -lGL; ./nBody
//To stop hit "control c" in the window you launched it from.
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "../cudaErrCheck.cuh"

#define N 8*8*8
#define BLOCK 256

#define XWindowSize 2500
#define YWindowSize 2500

#define DRAW 10
#define DAMP 1.0

#define DT 0.001
#define STOP_TIME 10.0

#define G 1.0
#define H 1.0

#define EYE 10.0
#define FAR 90.0

// Globals
float4 *p;
float3 *v, *f;
float4 *p_GPU0, *p_GPU1;
float rot=0.0; //to rotate cheerios

/*------------------------------------
|
|	DataStruct is to work on each GPU
|	
------------------------------------*/

struct DataStruct {
	int deviceID;
	int size;
	int offset;
	float4 *pos;
	float3 *vel;
	float3 *force;
};

void set_initial_conditions()
{
	p = (float4*)malloc(N*sizeof(float4));
	v = (float3*)malloc(N*sizeof(float3));
	f = (float3*)malloc(N*sizeof(float3));

	int i,j,k,num,particles_per_side;
    float position_start, temp;
    float initail_seperation;

	temp = pow((float)N,1.0/3.0) + 0.99999;
	particles_per_side = temp;
	printf("\n cube root of N = %d \n", particles_per_side);
    position_start = -(particles_per_side -1.0)/2.0;
	initail_seperation = 2.0;
	for(i=0; i<N; i++)
	{
		p[i].w = 1.0;
	}
	num = 0;
	for(i=0; i<particles_per_side; i++)
	{
		for(j=0; j<particles_per_side; j++)
		{
			for(k=0; k<particles_per_side; k++)
			{
			    if(N <= num) break;
				p[num].x = position_start + i*initail_seperation;
				p[num].y = position_start + j*initail_seperation;
				p[num].z = position_start + k*initail_seperation;
				v[num].x = 0.0;
				v[num].y = 0.0;
				v[num].z = 0.0;
				num++;
				
			}
		}
	}	
}

void draw_picture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	//gray white (powdered donut)
	//0.9955,0.8622,0.6711
	glColor3d(0.87,0.87,0.87);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(p[i].x, p[i].y, p[i].z);
		//make some cheerios 
		glRotatef(rot*i,p[i].x, p[i].y, p[i].z);
		glutSolidTorus(0.04,0.08,15,15);
		glPopMatrix();
	}
	rot+=0.1;
	
	glutSwapBuffers();
}
                                 
__device__ float3 getBodyBodyForce(float4 p0, float4 p1)
{
    float3 f;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz;
    float r = sqrt(r2);
    
    float force  = (G*p0.w*p1.w)/(r2) - (H*p0.w*p1.w)/(r2*r2);
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

__global__ void getForces(float4 *g_pos, float3 * force, int offset)
{
	int j,ii;
    float3 force_mag, forceSum;
    float4 posMe;
    __shared__ float4 shPos[BLOCK];
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    
    forceSum.x = 0.0;
	forceSum.y = 0.0;
	forceSum.z = 0.0;
		
	posMe.x = g_pos[id+offset].x;
	posMe.y = g_pos[id+offset].y;
	posMe.z = g_pos[id+offset].z;
	posMe.w = g_pos[id+offset].w;
	    
    for(j=0; j < gridDim.x*2; j++)
    {
    	shPos[threadIdx.x] = g_pos[threadIdx.x + blockDim.x*j];
    	__syncthreads();
   
		#pragma unroll 32
        for(int i=0; i < blockDim.x; i++)	
        {
        	ii = i + blockDim.x*j;
		    if(ii != id+offset && ii < N) 
		    {
		    	force_mag = getBodyBodyForce(posMe, shPos[i]);
			    forceSum.x += force_mag.x;
			    forceSum.y += force_mag.y;
			    forceSum.z += force_mag.z;
		    }
	   	}
	}
	if(id <N)
	{
	    force[id].x = forceSum.x;
	    force[id].y = forceSum.y;
	    force[id].z = forceSum.z;
    }
}

__global__ void moveBodies(float4 *g_pos, float4 *d_pos, float3 *vel, float3 * force, int offset)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if(id < N)
    {
	    vel[id].x += ((force[id].x-DAMP*vel[id].x)/d_pos[id].w)*DT;
	    vel[id].y += ((force[id].y-DAMP*vel[id].y)/d_pos[id].w)*DT;
	    vel[id].z += ((force[id].z-DAMP*vel[id].z)/d_pos[id].w)*DT;
	
		d_pos[id].x += vel[id].x*DT;
	    d_pos[id].y += vel[id].y*DT;
		d_pos[id].z += vel[id].z*DT;
		
		g_pos[id+offset].x = d_pos[id].x;
		g_pos[id+offset].y = d_pos[id].y;
		g_pos[id+offset].z = d_pos[id].z;
    }
}

void n_body()
{
	int deviceCount;
	ERROR_CHECK( cudaGetDeviceCount ( &deviceCount ) );
	p_GPU0 = (float4*)malloc(N*sizeof(float4));
	p_GPU1 = (float4*)malloc(N*sizeof(float4));



	DataStruct* dev = (DataStruct*)malloc(deviceCount*sizeof(DataStruct));
	
	for(int i = 0; i<deviceCount; i++)
	{
		cudaSetDevice(i);
		if(i==0)
		{
			ERROR_CHECK( cudaMalloc(&p_GPU0, N*sizeof(float4)) );
			ERROR_CHECK( cudaMemcpy(p_GPU0, p, N*sizeof(float4), cudaMemcpyHostToDevice) );
		}
		if(i==1)
		{
			ERROR_CHECK( cudaMalloc(&p_GPU1, N*sizeof(float4)) );
			ERROR_CHECK( cudaMemcpy(p_GPU1, p, N*sizeof(float4), cudaMemcpyHostToDevice) );
		}



		dev[i].deviceID = i;
		dev[i].size = N/deviceCount;
		dev[i].offset = i*N/deviceCount;
		ERROR_CHECK( cudaMalloc(&dev[i].pos, dev[i].size * sizeof(float4)) );
		ERROR_CHECK( cudaMalloc(&dev[i].vel, dev[i].size * sizeof(float3)) );
		ERROR_CHECK( cudaMalloc(&dev[i].force, dev[i].size * sizeof(float3)) );

		ERROR_CHECK( cudaMemcpy(dev[i].pos, p+dev[i].offset, dev[i].size * sizeof(float4), cudaMemcpyHostToDevice) );
		ERROR_CHECK( cudaMemcpy(dev[i].vel, v+dev[i].offset, dev[i].size * sizeof(float3), cudaMemcpyHostToDevice) );
		ERROR_CHECK( cudaMemcpy(dev[i].force, f+dev[i].offset, dev[i].size * sizeof(float3), cudaMemcpyHostToDevice) );
	}

	dim3 block(BLOCK);
	dim3 grid((N/deviceCount - 1)/BLOCK + 1);
	
	float dt;
	int   tdraw = 0; 
	float time = 0.0;
	float elapsedTime;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	dt = DT;
    
	while(time < STOP_TIME)
	{	
		for(int i = 0; i < deviceCount; i++)
		{
			float4 *temp;
			temp = i?p_GPU1:p_GPU0;
			cudaSetDevice( dev[i].deviceID );
			getForces<<<grid, block>>>(temp, dev[i].force, dev[i].offset);
			ERROR_CHECK( cudaPeekAtLastError() );
			moveBodies<<<grid, block>>>(temp, dev[i].pos, dev[i].vel, dev[i].force, dev[i].offset);
			ERROR_CHECK( cudaPeekAtLastError() );
		}

		cudaDeviceSynchronize();

		if(deviceCount > 1)
		{
			cudaSetDevice( 0 );
			ERROR_CHECK( cudaMemcpy(p_GPU1+dev[0].offset, dev[0].pos, dev[1].size*sizeof(float4), cudaMemcpyDeviceToDevice) );
			cudaSetDevice( 1 );
			ERROR_CHECK( cudaMemcpy(p_GPU0+dev[1].offset, dev[1].pos, dev[0].size*sizeof(float4), cudaMemcpyDeviceToDevice) );
		}

		cudaDeviceSynchronize();


		//To kill the draw comment out the next 7 lines.
		if(tdraw == DRAW) 
		{
			cudaSetDevice(0);
			ERROR_CHECK( cudaMemcpy(p, p_GPU0, N * sizeof(float4), cudaMemcpyDeviceToHost) );
			draw_picture();
			tdraw = 0;
		}
		tdraw++;
		
		time += dt;
	}
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\n\nGPU time = %3.1f milliseconds\n", elapsedTime);
}

void control()
{	
	set_initial_conditions();
	draw_picture();
    n_body();
	
	printf("\n DONE \n");
}

void Display(void)
{
	gluLookAt(EYE, EYE, EYE, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	control();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, FAR);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("2 Body 3D");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
