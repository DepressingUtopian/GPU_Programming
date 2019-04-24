
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <clocale>
#include <stdio.h>

#include <cuda_occupancy.h>


const int sizeShared = 32;
//const int arraySize = 30484848;
//Макрос для обработки ошибок
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

__global__ void MatrixInicialization(float *_matrix)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int N = blockDim.x * gridDim.x;

	_matrix[i + j * N] = (float)(i + j * N);
}

//NVIDIA DOCS 1
__global__ void coalescedMultiply(float *a, float* b, float *c,
	int N)
{
	__shared__ float aTile[sizeShared][sizeShared];

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	aTile[threadIdx.y][threadIdx.x] = a[row*sizeShared + threadIdx.x];
	for (int i = 0; i < sizeShared; i++) {
		sum += aTile[threadIdx.y][i] * b[i*N + col];
	}
	c[row*N + col] = sum;
}
//NVIDIA DOCS 1
__global__ void sharedABMultiply(float *a, float* b, float *c,
	int N)
{
	__shared__ float aTile[sizeShared][sizeShared],
		bTile[sizeShared][sizeShared];
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	aTile[threadIdx.y][threadIdx.x] = a[row*sizeShared + threadIdx.x];
	bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N + col];
	__syncthreads();
	for (int i = 0; i < sizeShared; i++) {
		sum += aTile[threadIdx.y][i] * bTile[i][threadIdx.x];
	}
	c[row*N + col] = sum;
}
__global__ void MatrixTranspose(float *_matrix1, float *_matrix2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int N = blockDim.x * gridDim.x;

	
	_matrix1[j + i * N] = _matrix2[i + j * N];
}
__global__ void MatrixTranspose_with_SharedMemory(float *_matrix1, float *_matrix2)
{
	__shared__ float tempMemory[sizeShared][sizeShared];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int N = blockDim.x * gridDim.x;

	tempMemory[threadIdx.y][threadIdx.x] = _matrix2[i + j * N];
	//printf("%d \n", i + j * N);
	__syncthreads();

	i = threadIdx.x + blockIdx.x * blockDim.x;
	j = threadIdx.y + blockIdx.y * blockDim.y;
	
	_matrix1[i + j * N] = tempMemory[threadIdx.x][threadIdx.y];
}
__global__ void MatrixTranspose_with_SharedMemoryCoalising(float *_matrix1, float *_matrix2)
{
	__shared__ float tempMemory[sizeShared][sizeShared + 1]; //Решение коализии

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int N = blockDim.x * gridDim.x;

	tempMemory[threadIdx.y][threadIdx.x] = _matrix2[i + j * N];
	//printf("%d \n", i + j * N);
	__syncthreads();

	i = threadIdx.x + blockIdx.x * blockDim.x;
	j = threadIdx.y + blockIdx.y * blockDim.y;

	_matrix1[i + j * N] = tempMemory[threadIdx.x][threadIdx.y];
}
void ViewMatrix(float *a, int size);
void MatrixTest(int size, int countThread);

using namespace std;

int main()
{
	setlocale(LC_ALL, "Russian");
	
	MatrixTest(2048,32);
	cout << "33" << endl;
	return 0;
}

void MatrixTest(int size,int countThread)
{

	int gridSize = size / countThread;

	if (size % countThread)
	{
		cout << "Ошибка количество потоков должно быть четным" << endl;
	}
	if (countThread > size)
	{
		cout << "Слишком много потоков" << endl;
	}
	float *a = new float[size * size];
	float *b = new float[size * size];
	float *c = new float[size * size];


	float *dev_a = nullptr;
	float *dev_b = nullptr;
	float *dev_c = nullptr;

	cudaError_t cudaStatus;
	for(int i = 0;i < size * size;i++)	
		{
			a[i] = 1;
			b[i] = 1;
		}

	cudaStatus = cudaMalloc((void**)&dev_a, size * size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");

	}

	cudaStatus = cudaMemcpy(dev_a, a, size * size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}
	cudaStatus = cudaMalloc((void**)&dev_b, size * size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");

	}

	cudaStatus = cudaMemcpy(dev_b, b, size * size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}


	cudaStatus = cudaMalloc((void**)&dev_c, size * size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");

	}

	cudaStatus = cudaMemcpy(dev_c, c, size * size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}
	sharedABMultiply << <dim3(gridSize, gridSize), dim3(countThread, countThread) >> > (dev_a,dev_b,dev_c,size);
	cudaDeviceSynchronize();
	/*MatrixInicialization << <dim3(gridSize, gridSize), dim3(countThread, countThread) >> > (dev_a);
	MatrixTranspose_with_SharedMemory << < dim3(gridSize, gridSize), dim3(countThread, countThread) >> > (dev_c, dev_a);
	cudaDeviceSynchronize();
	MatrixInicialization << < dim3(gridSize, gridSize), dim3(countThread, countThread) >> > (dev_a);
	MatrixTranspose_with_SharedMemoryCoalising << < dim3(gridSize, gridSize), dim3(countThread, countThread) >> > (dev_c, dev_a);
	cudaDeviceSynchronize();*/

	cudaStatus = cudaMemcpy(a, dev_a, size * size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}
	cudaStatus = cudaMemcpy(b, dev_b,size * size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}
	cudaStatus = cudaMemcpy(c, dev_c, size * size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	ViewMatrix(c, size);
}

void ViewMatrix(float *a, int size)
{
	cout << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			if ((j + i * size) % size == 0)
				cout << a[j + i * size] << " ";
			else
				cout << "\t" << a[j + i * size] << " ";

		cout << endl;
	}

	cout << endl;
}