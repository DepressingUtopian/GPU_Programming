
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include <cstdlib>
#include <iostream>

#include <clocale>

//Макрос для обработки ошибок
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
//Константы

const int countThreads = 1024; //Кол во Нитей
const int countBlocks = 1;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size,unsigned int countBlock,unsigned int countThread);

static void RunVectorSumm(int sizeVector, int countBlock, int countThread);

//Функция выполнящаяся на GPU
__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	int x;
	setlocale(LC_ALL,"Russian");
	srand(time(NULL));
	//Цикл по размерности вектора от 10 до 24
	for (int i = 10; i < 24; i++)		
		RunVectorSumm(i, countBlocks, countThreads);
	
    return 0;
}

void RunVectorSumm(int sizeVector, int countBlock, int countThread)
{
	//Инициализация вектора
	int arraySize = sizeVector;
	int *a  = new int[arraySize];
	int *b = new int[arraySize];
	int *c = new int[arraySize];

	for (int i = 0;i < arraySize;i++)
	{
		a[i] = rand();
		b[i] = rand();
	}
	
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize,countBlock,countThread);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return;
	}
	//printf(" %d" + c[0]);
	
	/*printf("{");
	for (int i = 0; i < arraySize; i++)
	{
		printf(" %d" + c[i]);
	}
	printf("}");
	printf("\n");*/

	//Данный команды сбрасывают состояние устройств перед выходом для корректной работы инструментов профилирования и отслеживания.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return;
	}
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size, unsigned int countBlock, unsigned int countThread)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaCheckError(cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice));
    
	
	//Запуск счетчика Event
	cudaEventRecord(start, 0);
    
	//Вычисляем результаты на GPU countBlock - количество блоков(варпов) , countThread - количество нитей
    addKernel<<<countBlock, countThread >>>(dev_c, dev_a, dev_b);
	cudaEventRecord(stop, 0);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    

	cudaCheckError(cudaDeviceSynchronize());
 
	cudaCheckError(cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost));
   
	
	//Остановка счетчика Event
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << std::endl;
	printf("Время выполнения: %f ms\n", time);

	//Вывод результата вычислений
	/*
	printf("{");
	for (int i = 0; i < size; i++)
	{
		printf(" %d", a[i]);

	}
	printf("} + ");
	printf("{");
	for (int i = 0; i < size; i++)
	{

		printf(" %d", b[i]);

	}
	printf("} = ");
	printf("{");
	for (int k = 0; k < size; k++)
	{
		printf(" %d", c[k]);
	}
	printf("}");*/
	
Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
