
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <clocale>
#include <stdio.h>

#include <cuda_occupancy.h>



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

__global__ void MatrixInicializationBlock(float *_matrix)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int N = blockDim.x * gridDim.x;

	_matrix[i + j * N] = (float)(i + j * N);
}
					
__global__ void MatrixTranspose(float *_matrix1, float *_matrix2)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int N = blockDim.x * gridDim.x;

	_matrix1[j + i * N] = _matrix2[i + j * N];
}

void getViewGPUSpec();
void Occupancy(int N);
void ViewMatrix(float *a, int size);
void MatrixTest(int size,void* func);

using namespace std;

int main()
{
	setlocale(LC_ALL,"Russian");
	getViewGPUSpec();

	for (long long int i = 10; i < 10e4; i <<= 1)
	{
		Occupancy(i);
	}
	MatrixTest(20, MatrixInicialization);

    return 0;
}

void getViewGPUSpec()
{
	int count = 0;
	int dev;
	cudaDeviceProp prop;

	cudaError_t error_id = cudaGetDeviceCount(&count);//Получаем количество устройств GPU

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

		

		for (dev = 0; dev < count; ++dev) {

			cudaSetDevice(dev);
			cudaGetDeviceProperties(&prop, dev);
			cout << "Получение информации об устройстве : " << prop.name << endl;
			cout << "Объем глобальной памяти : " << prop.totalGlobalMem / 1024 / 1024 << " MB" << endl;
			cout << "Объем разделяемой памяти в одном блоке : " << prop.sharedMemPerBlock << " байт " << endl;
			cout << "Количество нитей в варпе  : " << prop.warpSize << endl;
			cout << "Количество мультипроцессоров  : " << prop.multiProcessorCount << endl;
			cout << "Объем константной памяти  : " << prop.totalConstMem << endl;
			cout << "Максимальное количество нитей на  мультипроцессор : " << prop.maxThreadsPerMultiProcessor << endl;
			cout << "Максимальное количество нитей на блок  : " << prop.maxThreadsPerBlock << endl;

			cout <<endl << "Теоретическая заполняемость о числа нитей в блоке" << endl;
			cout << "Оптимальное количество блоков в варпе :" << prop.maxThreadsPerMultiProcessor / prop.warpSize << endl;
			cout << "Оптимальное количество нитей в блоке :" << prop.maxThreadsPerMultiProcessor / prop.multiProcessorCount << endl;
		

		}
}

void Occupancy(int N)
{
	int blockSize = 0;
	int numBlocks = 0;
	int minGridSize = 0;
	int activeWarps = 0;
	int maxWarps = 0;

	int arraySize = N;
	int device;

	int gridSize;

	float *a = new float[arraySize];
	float *b = new float[arraySize];

	float *dev_a = 0;
	float *dev_b = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);

	cudaDeviceProp prop;

	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");

	}
	cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");

	}

	cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");

	}

	cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}

	cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}


	cudaOccupancyMaxPotentialBlockSize(
		&minGridSize,
		&blockSize,
		(void*)MatrixInicialization,
		0,
		arraySize);

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocks,
		MatrixInicialization,
		blockSize,
		0);


	// Round up according to array size
	gridSize = (arraySize + blockSize - 1) / blockSize;



	//gTest2 <<<countBlocks, countThreads >>> (dev_b);
	//cudaDeviceSynchronize();

	activeWarps = numBlocks * blockSize / prop.warpSize;
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

	cout << "Метрика для N -мерного вектора" << endl;
//	cout << "Оптимальное количество варпов  для N = " << arraySize << " , = " <<  (double)gridSize / ((gridSize + blockSize - 1) / gridSize) << endl;
	cout << "Оптимальное количество блоков   для N = " << arraySize << " , = " << gridSize << endl;
	cout << "Оптимальное количество нитей   для N = " << arraySize << " , = " << blockSize << endl;
	std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
	std::cout << "Occupancy: " << (double)((blockSize * prop.multiProcessorCount) / (gridSize * maxWarps)) * 100 << "%" << std::endl;
	MatrixInicialization <<<gridSize, blockSize >>> (dev_a);
	cudaDeviceSynchronize();

Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
}

void getOccupancy(int N,int &_gridSize,int &_blockSize, void* func)
{
	int blockSize = 0;
	int numBlocks = 0;
	int minGridSize = 0;
	int activeWarps = 0;
	int maxWarps = 0;

	int arraySize = N;
	int device;

	int gridSize;


	cudaError_t cudaStatus;


	cudaOccupancyMaxPotentialBlockSize(
		&minGridSize,
		&blockSize,
		MatrixInicialization,
		0,
		N);

	// Round up according to array size
	gridSize = (N + blockSize - 1) / blockSize;

	_gridSize = gridSize;
	_blockSize = blockSize;
}
void MatrixTest(int size, void* func)
{

	int blockSize = 0;
	int gridSize = 0;

	dim3 grid(size, size);

	float *a = new float[size * size];
	float *b = new float[size * size];
	float *c = new float[size * size];


	float *dev_a = nullptr;
	float *dev_b = nullptr;
	float *dev_c = nullptr;

	cudaError_t cudaStatus;

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
	
	getOccupancy(size, gridSize, blockSize, func);
	

	MatrixInicialization <<<1, grid >>> (dev_a);
	MatrixInicializationBlock << <grid, 1 >> > (dev_b);
	MatrixTranspose << <1, grid >> > (dev_c,dev_a);
	cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(a,dev_a, size * size  * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");

	}
	cudaStatus = cudaMemcpy(b, dev_b, size * size * sizeof(float), cudaMemcpyDeviceToHost);
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

	ViewMatrix(c,size);
}

void ViewMatrix(float *a,int size)
{
	cout << endl;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
			if((j + i * size) % size == 0)
				cout << a[j + i * size] << " ";
			else
				cout << "\t" << a[j + i * size] << " ";

		cout << endl;
	}
				
	cout << endl;
}