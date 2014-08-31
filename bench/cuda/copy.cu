#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "util.h"

#define BLOCK_SIZE 512

//__global__ void copy(float * in, float * out, int size)
__global__ void copy(float * out)  
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	__syncthreads();
	int a_0 = tid * 0;
	int a_1 = tid * 1 + a_0;
	int a_2 = tid * 2 + a_1;
	int a_3 = tid * 3 + a_2;
	int a_4 = tid * 4 + a_3;
	int a_5 = tid * 5 + a_4;
	__syncthreads();
	
	int val = a_0 - a_1 + a_2 - a_3 + a_4 - a_5;
	out[tid] = val;
}

int main(int argc, char* argv[]) 
{
	bool verify = true;
	/* memory allocation*/	
	int w = 1024, h = 1024;
	int size = w * h;
	float * h_in = (float *)malloc(size * sizeof(float));
	float * h_out = (float *)malloc(size * sizeof(float));
	float * d_in, * d_out;
	cudaMalloc(&d_in, size * sizeof(float));
	cudaMalloc(&d_out, size*sizeof(float));
	
	/* transfer data from host(cpu) to device(gpu) */
	cudaMemcpy(d_in, h_in, size * sizeof(float), cudaMemcpyHostToDevice);

	/* kernel execution */
	int work_items = size;
	if(work_items % BLOCK_SIZE != 0)
	{
		work_items = size + (BLOCK_SIZE - size % BLOCK_SIZE);
	}
	//copy<<<work_items/BLOCK_SIZE, BLOCK_SIZE>>>(d_in, d_out, size);
	copy<<<work_items/BLOCK_SIZE, BLOCK_SIZE>>>(d_out);
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess)
	{
		printf("failure in kernel::%s\n", cudaGetErrorString(e));
		exit(1);
	}	
	cudaThreadSynchronize();

	/* transfer data from gpu to cpu */
	cudaMemcpy(h_out, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);

	if(verify)
	{
		verify_array<float>(h_in, h_out, size);
	}	
	
	cudaFree(d_in);
	cudaFree(d_out);
	if(h_in != 0) free(h_in);
	if(h_out != 0) free(h_out);
	
	return 0;
}

