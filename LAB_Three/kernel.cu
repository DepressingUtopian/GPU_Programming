
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <clocale>
#include <stdio.h>

//������ ��� ��������� ������
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
//���������

const int countThreads = 1024; //��� �� �����
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

	cudaError_t error_id = cudaGetDeviceCount(&count);//�������� ���������� ��������� GPU

	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",
			static_cast<int>(error_id), cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

		

		for (dev = 0; dev < count; ++dev) {

			cudaSetDevice(dev);
			cudaGetDeviceProperties(&prop, dev);
			cout << "��������� ���������� �� ���������� : " << prop.name << endl;
			cout << "����� ���������� ������ : " << prop.totalGlobalMem / 1024 / 1024 << " MB" << endl;
			cout << "����� ����������� ������ � ����� ����� : " << prop.sharedMemPerBlock << " ���� " << endl;
			cout << "���������� ����� � �����  : " << prop.warpSize << endl;
			cout << "���������� �����������������  : " << prop.multiProcessorCount << endl;
			cout << "����� ����������� ������  : " << prop.totalConstMem << endl;
			cout << "������������ ���������� ����������������� �� ���� : " << prop.maxThreadsPerMultiProcessor << endl;
			cout << "������������ ���������� ������� �� ����  : " << prop.maxThreadsPerBlock << endl;

		}
}

