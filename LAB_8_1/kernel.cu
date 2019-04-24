
#pragma comment (lib,"cublas.lib")

#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cublas.h>
#include <cublas_v2.h>



using namespace std;

#define N 1<<10
#define V 0.2
#define T 2
struct functor {
	const float koef;
	functor(float _koef) : koef(_koef) {}
	__host__ __device__ float operator()(float x, float y) { return koef * x + y; }
};
void saxpy(float _koef, thrust::device_vector<float> &x,
	thrust::device_vector<float> &y)
{
	functor func(_koef);
	thrust::transform(x.begin(), x.end(), y.begin(), y.begin(), func);

}


int main()
{
	setlocale(LC_ALL, "Russian");
	float Function[N];
	float FunctionData[N];
	cudaEvent_t start, stop;
	float *x = new float [N];
	float *y = new float [N];
	thrust::host_vector<float> cpumem1(N);
	thrust::host_vector<float> cpumem2(N);

	float *dev_x;
	float *dev_y;

	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = V * T;

	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc(&dev_x, N);
	cudaMalloc(&dev_y, N);

	for (int i = 0; i < N; i++) {
		FunctionData[i] = rand() % 100;
	}


	

	for (int i = 0; i < N; i++)
	{
		cpumem1[i] = FunctionData[i];

		(i - 1 >= 0) ? cpumem2[i] = FunctionData[i - 1] : cpumem2[i] = FunctionData[N - 1];
	}
	thrust::device_vector<float> gpumem1 = cpumem1;
	thrust::device_vector<float> gpumem2 = cpumem2;
	for (int i = 0; i < N; i++)
	{
		x[i] = cpumem1[i];
		y[i] = cpumem2[i];
	}
	cudaEventSynchronize(start);
	

	cudaEventRecord(start, 0);

	saxpy(V*T, gpumem1, gpumem1);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	std::cout << std::endl;

	printf("Thrust\n");
	printf("Время выполнения: %f ms\n", time);
	//for (int i = 0; i < N; i++)
	//	cout << gpumem1[i] << " ";

	cublasInit();

	cublasSetVector(N, sizeof(x[0]), x, 1, dev_x, 1);
	cublasSetVector(N, sizeof(y[0]), y, 1, dev_y, 1);

	cudaEventSynchronize(start);
	cudaEventRecord(start, 0);

	cublasSaxpy(handle, N, &alpha, dev_x, 1, dev_y, 1);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cublasGetVector(N, sizeof(y[0]), dev_y, 1, y, 1);
	cublasShutdown();

	std::cout << std::endl;
	printf("cuBLAS\n");
	printf("Время выполнения: %f ms\n", time);
	
	free(x);
	free(y);
	cudaFree(dev_x);
	cudaFree(dev_y);
	cublasDestroy(handle);
	return 0;
}

float TransportEquation()
{
	return 0;
}