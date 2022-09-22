#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>

__global__ void kernel() 
{ 
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  printf("%d ", idx + 1); 
}

void main()
{
  //Решение задачи на CPU
  printf("\n\n\n=====	CPU	=====\n\n");
  int start1, start2, time1, time2;
  start1 = clock();
  for (int i = 1; i != 25536; i++)
  {
  	printf("%d ", i);
  }
  time1 = clock() – start1;
  //Вызов ядра (решение задачи на GPU)
  	printf("=====	GPU	=====\n\n");
  start2 = clock();
  kernel<<<255, 257>>>();
  if (cudaGetLastError() != cudaSuccess)
  {
    printf("ERROR: kernel error.\n");
  }
  time2 = clock() – start2;
  char c;
  scanf("%c", &c);
  printf("\n\n\nTime CPU = %d\nTime GPU = %d\n", time1, time2);
}
