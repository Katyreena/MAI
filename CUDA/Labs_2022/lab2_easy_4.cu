#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <device_functions.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <typeinfo>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define f1(x) (x + x * x + x * x * x + x * x * x * x)
#define f2(x) (x * x - 2 * x - 7)
#define f3(x) x - 7

#define FUNC(x) f2(x)
#define SIGN(x) ((FUNC(x) >= eps)? 1 : (FUNC(x) <= -eps)? -1 : 0)

#define countThreads 1024
#define countBlocks 1
#define eps 0.001

__global__ void P_dichotomy_GPU(float* result)
{
	__shared__ int positiveHalfAxis[countThreads];
	__shared__ int negativeHalfAxis[countThreads];
	__shared__ bool flagStop;
	int cacheId = threadIdx.x;
	int f0 = SIGN(0);
	float pointX;
	flagStop = false;
	int round = -1;
	while (++round < 10000)
	{
		if (flagStop) return;
		pointX = eps * ((cacheId + 1) + round * countThreads);
		positiveHalfAxis[cacheId] = SIGN(pointX);
		negativeHalfAxis[cacheId] = SIGN(-pointX);
		if (positiveHalfAxis[cacheId] == 0)
		{
			*result = pointX;
			flagStop = true;
		}
		if (negativeHalfAxis[cacheId] == 0)
		{
			*result = -pointX;
			flagStop = true;
		}
		__syncthreads();
		if (cacheId == 0)
		{
			if (flagStop) return;
			for (int i = 0; i < countThreads; i++)
			{
				if (positiveHalfAxis[i] * f0 == -1)
				{
					*result = eps * ((i + 1) + round * countThreads - 0.5);
					flagStop = true;
					break;
				}
				if (negativeHalfAxis[i] * f0 == -1)
				{
					*result = -eps * ((i + 1) + round * countThreads - 0.5);
					flagStop = true;
					break;
				}
			}
		}
		__syncthreads();
		if (flagStop) return;
	}
}

void P_dichotomy_CPU(float* result)
{
	int f0 = SIGN(0);
	double a, b, c;
	a = -DBL_MAX;
	b = DBL_MAX;
	while (b - a > eps) 
	{
		c = (a + b) / 2;
		if (FUNC(b) * FUNC(c) < 0) a = c;
		else b = c;
	}
	*result = (a + b) / 2;
}

int main(int argc, char* argv[])
{
	float result;
	float* dev_result, *cpu_result;
	cpu_result = (float*)malloc(sizeof(float));
	if (cudaMalloc(&dev_result, sizeof(float)) != cudaSuccess)
	{
		printf("Error: cudaMalloc\n");
	}

	// Вызов функции на GPU
	cudaEvent_t start_gpu, stop_gpu;
	float gpuTime = 0.0f;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
	cudaEventRecord(start_gpu, 0);
	P_dichotomy_GPU<<<countBlocks, countThreads>>>(dev_result);
	cudaEventRecord(stop_gpu, 0);
	cudaEventSynchronize(stop_gpu);
	cudaEventElapsedTime(&gpuTime, start_gpu, stop_gpu);

	// Вызов функции на CPU
	clock_t start_cpu, end_cpu;
	start_cpu = clock();
	P_dichotomy_CPU(cpu_result);
	end_cpu = clock();
	double cpuTime = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
	cpuTime *= 1000;

	// Вывод времени
	printf("Time CPU = %lf millseconds\n", cpuTime);
	printf("Time GPU = %f millseconds\n\n", gpuTime);

	if (cudaGetLastError() != cudaSuccess)
	{
		printf("Error: Kernal\n");
	}
	if (cudaMemcpy(&result, dev_result, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Error: cudaMemcpy2\n");
	}

	// Вывод
	printf("Answer GPU = %f\n", result);
	printf("Answer CPU = %f\n", *cpu_result);

	// Очищение памяти
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(stop_gpu);
}
