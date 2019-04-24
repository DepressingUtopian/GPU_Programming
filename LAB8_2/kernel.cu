#pragma comment (lib,"cufft.lib")


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cufft.h>
#include <stdio.h>
#include <malloc.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <list>
#include <stdlib.h>


#define NX 64
#define BATCH 1
#define pi 3.141592

using namespace std;

int main()
{
	string line;
	cufftHandle plan;
	cufftComplex *cpu_data;
	cufftComplex *gpu_data;
	std::vector<string> commands;
	vector<vector<string>> DATA;
	ifstream in("./data.txt");
	cufftComplex *data_h;

	if (in.is_open())
	{
		while (getline(in, line))
		{
			
			std::string buffer = "";      //буфферная строка
			for (int i = 0; i < line.size(); i++) {
				if (line[i] != ' ') {      // "—" сплиттер
					buffer += line[i];
				}
				else {
					if(buffer !="")
						commands.push_back(buffer);
					buffer = "";
				}
				if(i + 1 == line.size())
					commands.push_back(buffer);

			}
			if (commands.size() != 0)
			{
				DATA.push_back(commands);
				commands.clear();
			}
		}
	}
	in.close();

	cudaMalloc((void**)&gpu_data, sizeof(cufftComplex) *  DATA.size() * BATCH);
	data_h = (cufftComplex*)calloc(DATA.size(), sizeof(cufftComplex));
	cpu_data = new cufftComplex[DATA.size() * BATCH];
	for (int i = 0; i < DATA.size() * BATCH; i++)
	{
		cpu_data[i].x = stof(DATA[i][2]);
		cpu_data[i].y = stof(DATA[i][3]);
	}
	cudaMemcpy(gpu_data, cpu_data, sizeof(cufftComplex) *  DATA.size() * BATCH, cudaMemcpyHostToDevice);

	if (cufftPlan1d(&plan, DATA.size() * BATCH, CUFFT_C2C, BATCH) != CUFFT_SUCCESS)
	{
		cerr << "ERROR cufftPlan1d" << endl;
		return -1;
	}
	if (cufftExecC2C(plan, gpu_data, gpu_data, CUFFT_FORWARD) != CUFFT_SUCCESS)
	{
		cerr << "ERROR cufftPlan1d" << endl;
		return -1;
	}
	if (cudaDeviceSynchronize() != CUFFT_SUCCESS)
	{
		cerr << "ERROR cufftPlan1d" << endl;
		return -1;
	}

	cudaMemcpy(data_h,gpu_data, DATA.size() * sizeof(cufftComplex), cudaMemcpyDeviceToHost);

	for (int i = 0; i < DATA.size(); i++)
		printf("%g\t%g\n",data_h[i].x,data_h[i].y);
   //Проверку закономерности втыкнуть тут!
	cufftDestroy(plan);
	cudaFree(gpu_data);
	free(data_h);
	free(cpu_data);

    return 0;
}
