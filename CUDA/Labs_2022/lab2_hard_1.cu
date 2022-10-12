#ifndef __CUDACC__ 
#define __CUDACC__
#endif

#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define COUNT_THREADS ARR_ROW
#define ARR_ROW 5
#define ARR_COLUMN 3
#define ARR_SIZE (ARR_ROW * ARR_COLUMN)

#define COMPARE(x, y) ((x.Re == y.Re) ? ((x.Im >= y.Im) ? x : y) : ((x.Re > y.Re) ? x : y))
#define COMPARE_BOOL(x, y) ((x.Re == y.Re) ? ((x.Im >= y.Im) ? 1 : 0) : ((x.Re > y.Re) ? 1 : 0))

struct Complex
{
	float Re;
	float Im;
};

__global__ void find_max_GPU(Complex* matrix, Complex* arr_max, int* mutex)
{
	Complex* str = matrix + ARR_COLUMN * blockIdx.x;
	Complex* max = arr_max + blockIdx.x;
	Complex tmp = { 0, 0 };
	unsigned index = threadIdx.x;
	unsigned move = blockDim.x;
	unsigned addition = 0;
	__shared__ Complex cache[COUNT_THREADS];
	while (index + addition < ARR_COLUMN)
	{
		tmp = COMPARE(tmp, str[index + addition]);
		addition += move;
	}
	cache[threadIdx.x] = tmp;
	__syncthreads();
	unsigned i = blockDim.x / 2;
	while (i != 0)
	{
		if (threadIdx.x < i)
			cache[threadIdx.x] = COMPARE(cache[threadIdx.x], cache[threadIdx.x + i]);
		__syncthreads();
		i /= 2;
	}
	if (threadIdx.x == 0)
	{
		while (atomicCAS(mutex, 0, 1) != 0);
		*max = COMPARE((*max), cache[0]);
		atomicExch(mutex, 0);
	}
}

void find_max_CPU(Complex* arr, Complex* max)
{
	for (int i = 0; i != ARR_ROW; i++)
	{
		Complex tmp_max = arr[0];
		for (int j = 0; j != ARR_COLUMN; j++)
			if (COMPARE_BOOL(arr[i * ARR_COLUMN + j], tmp_max))
				tmp_max = arr[i * ARR_COLUMN + j];
		max[i] = tmp_max;
	}
}

void main()
{
	Complex* cpu_arr, * dev_arr, * max, * dev_max;
	int* mutex, * dev_mutex;
	max = (Complex*)malloc(ARR_ROW * sizeof(Complex));
	mutex = (int*)malloc(sizeof(int));
	cpu_arr = (Complex*)malloc(ARR_SIZE * sizeof(Complex));

	// Генерация матрицы
	printf("-----	Array:	-----\n");
	for (int i = 0; i != ARR_SIZE; i++)
	{
		cpu_arr[i].Re = rand() % 100;
		cpu_arr[i].Im = rand() % 100;
		printf("Re = %f; Im = %f\t", cpu_arr[i].Re, cpu_arr[i].Im);
		if ((i + 1) % ARR_COLUMN == 0 && i != 0)
		{
			printf("\n");
		}
	}
	if (cudaMalloc(&dev_arr, ARR_SIZE * sizeof(Complex)) != cudaSuccess)
	{
		printf("Error: cudaMalloc");
	}
	if (cudaMemcpy(dev_arr, cpu_arr, ARR_SIZE * sizeof(Complex), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("Error: cudaMemcpy!");
	}
	if (cudaMalloc(&dev_max, ARR_ROW * sizeof(Complex)) != cudaSuccess)
	{
		printf("Error: cudaMalloc");
	}
	if (cudaMalloc(&dev_mutex, sizeof(int)) != cudaSuccess)
	{
		printf("Error: cudaMalloc");
	}
	
	// Вызов функции на GPU
	cudaEvent_t start_gpu, stop_gpu;
	float gpuTime = 0.0f;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);
	cudaEventRecord(start_gpu, 0);
	find_max_GPU <<<ARR_ROW, ARR_COLUMN>>> (dev_arr, dev_max, dev_mutex);
	cudaEventRecord(stop_gpu, 0);
	cudaEventSynchronize(stop_gpu);
	cudaEventElapsedTime(&gpuTime, start_gpu, stop_gpu);

	// Вызов функции на CPU
	clock_t start_cpu, end_cpu;
	start_cpu = clock();
	find_max_CPU(cpu_arr, max);
	end_cpu = clock();
	double cpuTime = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
	cpuTime *= 1000;

	// Вывод времени
	printf("\n\nTime CPU = %lf millseconds\n", cpuTime);
	printf("Time GPU = %f millseconds\n\n", gpuTime);
	
	if (cudaGetLastError() != cudaSuccess)
	{
		printf("Error: find_max");
	}
	if (cudaMemcpy(max, dev_max, ARR_ROW * sizeof(Complex), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Error: cudaMemcpy!");
	}
	
	// Вывод
	printf("-----	CPU result	-----\n");
	for (int i = 0; i != ARR_ROW; i++)
	{
		printf("%d Answer: Re = %f; Im = %f\n", i, max[i].Re, max[i].Im);
	}
	printf("\n-----	GPU result	-----\n");
	for (int i = 0; i != ARR_ROW; i++)
	{
		printf("%d Answer: Re = %f; Im = %f\n", i, max[i].Re, max[i].Im);
	}

	// Очищение памяти
	cudaEventDestroy(start_gpu);
	cudaEventDestroy(stop_gpu);
	cudaFree(dev_arr);
	cudaFree(dev_max);
	cudaFree(dev_mutex);
	free(cpu_arr);
	free(max);
	free(mutex);
}
