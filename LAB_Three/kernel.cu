
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <clocale>
#include <stdio.h>

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

void getViewGPUSpec();

using namespace std;

int main()
{
	setlocale(LC_ALL,"Russian");
	getViewGPUSpec();
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
			cout << "Максимальное количество мультипроцессоров на нить : " << prop.maxThreadsPerMultiProcessor << endl;
			cout << "Максимальное количество потоков на блок  : " << prop.maxThreadsPerBlock << endl;

		}
}

