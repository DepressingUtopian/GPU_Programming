
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <stdio.h>
#include <cstdlib>
#include <iostream>

#include <clocale>

float cuda_memory_malloc_test(int size, bool up);
float cuda_alloc_memory_malloc_test(int size, bool up);
//Макрос для обработки ошибок
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
//Константы
int countThreads = 1024; //Кол во Нитей
int countBlocks = 1000;

cudaError_t addWithCuda(long  long int *c, const long  long int *a, const long  long int *b, long  long int size, unsigned int countBlock, unsigned int countThread);

static void RunVectorSumm(long  long int sizeVector, int countBlock, int countThread);

//Функция выполнящаяся на GPU
__global__ void addKernel(long  long int *c, const  long  long int *a, const  long  long int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

const int SIZE = (10 * 1024 * 1024);

int main()
{
	int x;
	setlocale(LC_ALL, "Russian");
	srand(time(NULL));

	float elapsedTime;
	float MB = (float)100 * SIZE * sizeof(int)/1024/1024;
	
	elapsedTime = cuda_memory_malloc_test(SIZE, true);
	printf("Время без использования блокируемых страниц памяти при копировании на GPU: %3.5f ms\n",elapsedTime);
	printf("\tМБ/с при копировании на GPU %3.1f\n", MB/(elapsedTime/1000));
	elapsedTime = cuda_memory_malloc_test(SIZE, false);
	printf("Время без использования блокируемых страниц памяти при копировании на CPU: %3.5f ms\n", elapsedTime);
	printf("\tМБ/с при копировании на CPU %3.1f\n", MB / (elapsedTime / 1000));
	
	elapsedTime = cuda_alloc_memory_malloc_test(SIZE, true);
	printf("Время c использованием блокируемых страниц памяти при копировании на GPU: %3.5f ms\n", elapsedTime);
	printf("\tМБ/с при копировании на GPU %3.1f\n", MB / (elapsedTime / 1000));
	elapsedTime = cuda_alloc_memory_malloc_test(SIZE, false);
	printf("Время с использованием блокируемых страниц памяти при копировании на CPU: %3.5f ms\n", elapsedTime);
	printf("\tМБ/с при копировании на CPU %3.1f\n", MB / (elapsedTime / 1000));

		std::cout << "Размерность: " << 100 << " Блоков: " << 10 << " Нитей: " << 10;
		countBlocks = (100 + countThreads - 1) / countThreads;
		RunVectorSumm(100, countBlocks, countThreads);



	return 0;
}

void RunVectorSumm(long  long int sizeVector, int countBlock, int countThread)
{
	//Инициализация вектора
	cudaStream_t stream0, stream1;
	long  long int arraySize = sizeVector;
	long  long int *a;
	long  long int *b;
	long  long int *c;


	cudaHostAlloc((void**)&a, sizeVector * sizeof(*a), cudaHostAllocDefault);
	cudaHostAlloc((void**)&b, sizeVector * sizeof(*b), cudaHostAllocDefault);
	cudaHostAlloc((void**)&c, sizeVector * sizeof(*c), cudaHostAllocDefault);

	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	for (int i = 0; i < arraySize; i++)
	{
		a[i] = rand();
		b[i] = rand();
	}

	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize, countBlock, countThread);
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

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(long  long int *c, const long  long int *a, const long  long int *b, long  long int size, unsigned int countBlock, unsigned int countThread)
{
	long  long int *dev_a0 = 0;
	long  long int *dev_b0 = 0;
	long  long int *dev_c0 = 0;

	long  long int *dev_a1 = 0;
	long  long int *dev_b1 = 0;
	long  long int *dev_c1 = 0;

	cudaStream_t stream0, stream1;
	cudaError_t cudaStatus;
	cudaEvent_t start, stop;


	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);
	
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
	cudaStatus = cudaMalloc((void**)&dev_c0, size * sizeof(long  long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a0, size * sizeof(long  long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b0, size * sizeof(long  long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_c1, size * sizeof(long  long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a1, size * sizeof(long  long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b1, size * sizeof(long  long int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);

	for (int i = 0; i < size - 1; i+=2)
	{
		//printf("\n %d",i);
		//printf("\n %d", i + 1);
		cudaStatus = cudaMemcpy(dev_a0, a + i, 1 * sizeof(long  long int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_a1, a + i + 1, 1 * sizeof(long  long int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_b0, b + i, 1 * sizeof(long  long int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(dev_b1, b + i + 1, 1 * sizeof(long  long int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		addKernel << <1,1, 1,stream0 >> > (dev_c0, dev_a0, dev_b0);
		addKernel << <1,1, 1, stream1 >> > (dev_c1, dev_a1, dev_b1);
		
		cudaMemcpyAsync(c + i, dev_c0, 1 * sizeof(long long int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(c + i + 1, dev_c1, 1 * sizeof(long long int), cudaMemcpyDeviceToHost, stream1);
	}
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	//Запуск счетчика Event
	
	//Вычисляем результаты на GPU countBlock - количество блоков(варпов) , countThread - количество нитей
	
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << std::endl;
	printf("Время выполнения: %f ms\n", time);
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}


	cudaCheckError(cudaDeviceSynchronize());

	cudaCheckError(cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(long  long int), cudaMemcpyDeviceToHost));


	//Остановка счетчика Event


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
	cudaFree(dev_c0);
	cudaFree(dev_a0);
	cudaFree(dev_b0);
	cudaFree(dev_c1);
	cudaFree(dev_a1);
	cudaFree(dev_b1);

	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);

	return cudaStatus;
}
float cuda_memory_malloc_test(int size,bool up)
{
	cudaEvent_t start, stop;
	cudaError_t cudaStatus;

	int *a, *dev_a;
	float elapsedTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	a = (int*)malloc(size * sizeof(*a));
	cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));
	

	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (up)
		{
			cudaMemcpy(dev_a, a, size * sizeof(*dev_a), cudaMemcpyHostToDevice);
		}
		else
			cudaMemcpy(a, dev_a, size * sizeof(*dev_a), cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);


	free(a);

	cudaFree(dev_a);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}
float cuda_alloc_memory_malloc_test(int size, bool up)
{
	cudaEvent_t start, stop;
	int *a, *dev_a;
	float elapsedTime = 0.0f;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)&dev_a, size * sizeof(*dev_a));
	cudaHostAlloc((void**)&a, size * sizeof(*a),cudaHostAllocDefault);

	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);

	for (int i = 0; i < 100; i++) {
		if (up)
		{
			cudaMemcpy(dev_a, a, size * sizeof(*a), cudaMemcpyHostToDevice);
		}
		else
			cudaMemcpy(a, dev_a, size * sizeof(*a), cudaMemcpyDeviceToHost);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaFree(dev_a);
	cudaFreeHost(a);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return elapsedTime;
}