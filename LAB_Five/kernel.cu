
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define M_PI 3.14159265358979323846
#define COEF 48
#define VERTCOUNT COEF*COEF*2//-(COEF-1)*2
#define RADIUS 10.0f
#define FGSIZE 20
#define FGSHIFT FGSIZE/2
#define IMIN(A,B) (A<B?A:B)
#define THREADSPERBLOCK 256
#define BLOCKSPERGRID IMIN(32,(VERTCOUNT+THREADSPERBLOCK-1)/THREADSPERBLOCK)

typedef float(*ptr_f)(float, float, float);

struct Vertex
{
	float x, y, z;
};

__constant__ Vertex vert[VERTCOUNT];
Vertex vert2[VERTCOUNT];

texture<float, 3, cudaReadModeElementType> df_tex;
cudaArray* df_Array = 0;

float interpolition();
float lerp(float x, float x1, float x2, float q00, float q01);
float biLerp(float x, float y, float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2);
float triLerp(float x, float y, float z, float q000, float q001, float q010, float q011, float q100, float q101, float q110, float q111, float x1, float x2, float y1, float y2, float z1, float z2);
float func(float x, float y, float z);

__device__ bool GetBit(unsigned int i, unsigned int position)
{
	return (bool(1 << position) & i);
}


float func(float x, float y, float z)
{
	return (0.5*sqrtf(15.0 / M_PI))*(0.5*sqrtf(15.0 / M_PI))*z*z*y*y*sqrtf(1.0f - z * z / RADIUS / RADIUS) / RADIUS / RADIUS / RADIUS / RADIUS;
}


__device__ float func2(float x, float y, float z)
{
	return (0.5*sqrtf(15.0 / M_PI))*(0.5*sqrtf(15.0 / M_PI))*z*z*y*y*sqrtf(1.0f - z * z / RADIUS / RADIUS) / RADIUS / RADIUS / RADIUS / RADIUS;
}
__global__ void kernel(float *a)
{
	__shared__ float cache[THREADSPERBLOCK];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	float x = vert[tid].x + FGSHIFT + 0.5f;
	float y = vert[tid].y + FGSHIFT + 0.5f;
	float z = vert[tid].z + FGSHIFT + 0.5f;
	cache[cacheIndex] = tex3D(df_tex, z, y, x);

	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cacheIndex < s)
			cache[cacheIndex] += cache[cacheIndex + s];
		__syncthreads();
	}

	if (cacheIndex == 0)
		a[blockIdx.x] = cache[0];
}
__global__ void kernel2(float *a)
{
	__shared__ float cache[THREADSPERBLOCK];
	
	int cacheIndex = threadIdx.x;

	cache[cacheIndex] = interpolition();

	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (cacheIndex < s)
			cache[cacheIndex] += cache[cacheIndex + s];
		__syncthreads();
	}

	if (cacheIndex == 0)
		a[blockIdx.x] = cache[0];
}
__device__ float interpolition()
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	Vertex interPoint[2]; 

	Vertex mainPoint; //Искомая точка

	//Получаем координатные значения для точек
	for (int i = 0; i < 2; i++) {
		int iphi = (tid + i) / COEF;
		int ipsi = (tid + i) % COEF;
		float phi = iphi * M_PI / COEF;
		float psi = ipsi * M_PI / COEF;
		interPoint[i].x = RADIUS * sinf(psi) * cosf(phi);
		interPoint[i].y = RADIUS * sinf(psi) * sinf(phi);
		interPoint[i].z = RADIUS * cosf(psi);
	}

	//Получение координат точки в середине куба
	mainPoint.x = (interPoint[0].x + interPoint[1].x) / 2.0f;
	mainPoint.y = (interPoint[0].y + interPoint[1].y) / 2.0f;
	mainPoint.z = (interPoint[0].z + interPoint[1].z) / 2.0f;

	double koef = (interPoint[1].x - interPoint[0].x) * (interPoint[1].y - interPoint[0].y) * (interPoint[1].z - interPoint[0].z);
	float summator = 0;

	if (koef == 0.0f)
		return 0.0f;
	for (int i = 0; i < 8; i++)
	{		
		
		float funcValue = func2(interPoint[(int)GetBit(i,0)].x, interPoint[(int)GetBit(i, 1)].y, interPoint[(int)GetBit(i, 2)].z);
		float xi = GetBit(i, 0) ?
			mainPoint.x - interPoint[0].x : interPoint[1].x - mainPoint.x;
		float yi = GetBit(i, 1) ?
			mainPoint.y - interPoint[0].y : interPoint[1].y - mainPoint.y;
		float zi = GetBit(i, 2) ?
			mainPoint.z - interPoint[0].z : interPoint[1].z - mainPoint.z;
		funcValue *= xi * yi * zi;
		
		summator += funcValue;
	}
	
	return summator / koef;
}
float lerp(float x, float x1, float x2, float q00, float q01) {
	return ((x2 - x) / (x2 - x1)) * q00 + ((x - x1) / (x2 - x1)) * q01;
}

float biLerp(float x, float y, float q11, float q12, float q21, float q22, float x1, float x2, float y1, float y2) {
	float r1 = lerp(x, x1, x2, q11, q21);
	float r2 = lerp(x, x1, x2, q12, q22);

	return lerp(y, y1, y2, r1, r2);
}

float triLerp(float x, float y, float z, float q000, float q001, float q010, float q011, float q100, float q101, float q110, float q111, float x1, float x2, float y1, float y2, float z1, float z2) {
	float x00 = lerp(x, x1, x2, q000, q100);
	float x10 = lerp(x, x1, x2, q010, q110);
	float x01 = lerp(x, x1, x2, q001, q101);
	float x11 = lerp(x, x1, x2, q011, q111);
	float r0 = lerp(y, y1, y2, x00, x01);
	float r1 = lerp(y, y1, y2, x10, x11);

	return lerp(z, z1, z2, r0, r1);
}

void calc_f(float *arr_f, int x_size, int y_size, int z_size, ptr_f f)
{
	for (int x = 0; x < x_size; ++x)
		for (int y = 0; y < y_size; ++y)
			for (int z = 0; z < z_size; ++z)
				arr_f[z_size * (x * y_size + y) + z] = f(x - FGSHIFT, y - FGSHIFT, z - FGSHIFT);
}

float check(Vertex *v, ptr_f f)
{
	float sum = 0.0f;
	for (int i = 0; i < VERTCOUNT; ++i)
		sum += f(v[i].x, v[i].y, v[i].z);

	return sum;
}

void init_vertexes()
{
	Vertex *temp_vert = (Vertex *)malloc(sizeof(Vertex) * VERTCOUNT);
	int i = 0;
	for (int iphi = 0; iphi < 2 * COEF; ++iphi)
	{
		for (int ipsi = 0; ipsi < COEF; ++ipsi, ++i)
		{
			float phi = iphi * M_PI / COEF;
			float psi = ipsi * M_PI / COEF;
			temp_vert[i].x = RADIUS * sinf(psi) * cosf(phi);
			temp_vert[i].y = RADIUS * sinf(psi) * sinf(phi);
			temp_vert[i].z = RADIUS * cosf(psi);
		}
	}
	printf("sumcheck = %f\n", check(temp_vert, &func)*M_PI*M_PI / COEF / COEF);
	cudaMemcpyToSymbol(vert, temp_vert, sizeof(Vertex) * VERTCOUNT, 0, cudaMemcpyHostToDevice);
	cudaMemcpy(vert2, temp_vert, sizeof(Vertex) * VERTCOUNT,cudaMemcpyHostToDevice);
	free(temp_vert);
}

void init_texture(float *df_h)
{
	const cudaExtent volumeSize = make_cudaExtent(FGSIZE, FGSIZE, FGSIZE);
	cudaChannelFormatDesc  channelDesc = cudaCreateChannelDesc<float>();
	cudaMalloc3DArray(&df_Array, &channelDesc, volumeSize);
	cudaMemcpy3DParms  cpyParams = { 0 };
	cpyParams.srcPtr = make_cudaPitchedPtr((void*)df_h, volumeSize.width * sizeof(float), volumeSize.width, volumeSize.height);
	cpyParams.dstArray = df_Array;
	cpyParams.extent = volumeSize;
	cpyParams.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&cpyParams);
	df_tex.normalized = false;
	df_tex.filterMode = cudaFilterModeLinear;
	df_tex.addressMode[0] = cudaAddressModeClamp;
	df_tex.addressMode[1] = cudaAddressModeClamp;
	df_tex.addressMode[2] = cudaAddressModeClamp;
	cudaBindTextureToArray(df_tex, df_Array, channelDesc);
}

void release_texture()
{
	cudaUnbindTexture(df_tex);
	cudaFreeArray(df_Array);
}

int main(void)
{
	init_vertexes();

	float *arr = (float *)malloc(sizeof(float) * FGSIZE * FGSIZE * FGSIZE);
	calc_f(arr, FGSIZE, FGSIZE, FGSIZE, &func);
	init_texture(arr);

	float *sum = (float*)malloc(sizeof(float) * BLOCKSPERGRID);
	float *sum_dev;
	cudaMalloc((void**)&sum_dev, sizeof(float) * BLOCKSPERGRID);

	//Интерполяция сферы с использованием текстурной и константной памяти
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	kernel << <BLOCKSPERGRID, THREADSPERBLOCK >> > (sum_dev);
	cudaThreadSynchronize();
	cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID, cudaMemcpyDeviceToHost);
	float s = 0.0f;
	for (int i = 0; i < BLOCKSPERGRID; ++i)
		s += sum[i];
	printf("sum = %f\n", s*M_PI*M_PI / COEF / COEF);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time: %3.1f ms\n", elapsedTime);

	s = 0.0f;
	//Интерполяция сферы без использованием текстурной и константной памяти
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	kernel2 << <BLOCKSPERGRID, THREADSPERBLOCK >> > (sum_dev);
	cudaThreadSynchronize();
	cudaMemcpy(sum, sum_dev, sizeof(float) * BLOCKSPERGRID, cudaMemcpyDeviceToHost);
	for (int i = 0; i < BLOCKSPERGRID; ++i)
		s += sum[i];
	printf("sum 2= %f\n", s*M_PI*M_PI / COEF / COEF);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	//float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time: %3.1f ms\n", elapsedTime);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(sum_dev);
	free(sum);
	release_texture();
	free(arr);

	return 0;
}