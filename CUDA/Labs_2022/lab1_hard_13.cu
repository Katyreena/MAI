#define __CUDACC__
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define THREADS = 256;
#define BLOCKS = 64;

__global__ void StandartDeviation(float* numbers, float* result, int n, float x)
{
	__shared__ float cache[THREADS];
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int cacheId = threadIdx.x;
	int totalNumberOfThreads = gridDim.x * blockDim.x;
	float tmp = 0;
	while (idx < n)
	{
		tmp += (sample[idx] - x) * (sample[idx] - x);
		idx += totalNumberOfThreads;
	}

	cache[cacheId] = tmp;
	__syncthreads();
	int iter = blockDim.x / 2;
	while (iter != 0)
	{
		if (cacheId < iter) cache[cacheId] += cache[cacheId + iter];
		__syncthreads();
		iter /= 2;
	}

	if (cacheId == 0) result[blockIdx.x] = cache[0];
}

void main()
{
	int N;
	printf("Print count of sample: \n");
	while (scanf("%d", &N) != 1)
	{
		printf("Incorrect! Try again: ");
		while (getchar() != '\n');
	}

	float x = 0;
	float* a, *b, *c_result;
	float *dev_sample, *dd, *dev_result;
	a = (float*)malloc(N * sizeof(float));
	b = (float*)malloc(N * sizeof(float));
	c_result = (float*)malloc(BLOCKS * sizeof(float));
	if (cudaMalloc(&dev_sample, N * sizeof(float)) != cudaSuccess) printf("Error: cudaMalloc");
	if (cudaMalloc(&dd, N * sizeof(float)) != cudaSuccess) printf("Error: cudaMalloc");
	if (cudaMalloc(&dev_result, BLOCKS * sizeof(float)) != cudaSuccess) printf("Error: cudaMalloc");
	printf("Print %d numbers: \n", N);
	for (int i = 0; i != N; i++)
	{
		float tmp = 0;
		while ((scanf("%f", &tmp)) != 1)
		{
			printf("Incorrect! Try again: ");
			while (getchar() != '\n');
		}
		x += tmp;
		a[i] = tmp;
	}

	if (cudaMemcpy(dev_sample, a, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) printf("Error: cudaMemcpy!");
	if (cudaMemcpy(dd, b, N * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) printf("Error: cudaMemcpy!");
	StandartDeviation<<<BLOCKS, THREADS>>>(dev_sample, dev_result, N, (float)x/n);
	if (cudaGetLastError() != cudaSuccess) printf("Error: sumOfSquares");
	if (cudaMemcpy(partial_c, dev_partial_c, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost) != cudaSuccess) printf("Error: cudaMemcpy");
	if (cudaGetLastError() != cudaSuccess) printf("Error: sumOfSquares");
	if (cudaMemcpy(c_result, dev_result, sizeof(float) * BLOCKS, cudaMemcpyDeviceToHost) != cudaSuccess) printf("Error: cudaMemcpy");
	float c = 0;
	for (int i = 0; i != BLOCKS; i++) c += c_result[i];
	printf("Result: %f", c);
}
