#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CLHelper.h"
#include "util.h"
#include <string.h>

#ifndef TIME
#define TIME
#endif

#ifndef VARIFY
#define VARIFY
#endif


#define BLOCK_SIZE 16

void mm_1(cl_mem d_A, cl_mem d_B, cl_mem d_C, int wData, int hData);
void mm_2(cl_mem d_A, cl_mem d_B, cl_mem d_C, int wData, int hData);
void CPURun(float * A, float * B, float * refC, int wData, int hData);

int main(int argc, char ** argv)
{
	_clInit(0, "cpu", 0);
	uint iter = 100;
	
	_clRelease();

	return 1;
}

/*
	matrix tranpose on CPU as reference
*/
void CPURun(float * A, float * B, float * refC, int wData, int hData)
{
    for (unsigned int i = 0; i < hData; ++i)
    {
        for (unsigned int j = 0; j < wData; ++j)
        {
            float sum = 0;
            for (unsigned int k = 0; k < wData; ++k)
            {
                float a = A[i * wData + k];
                float b = B[k * wData + j];
                sum += a * b;
            }
            refC[i * wData + j] = (float)sum;
        }
    }
	return ;
}

/*
	Impl. of MM on GPUs from NV SDK
*/

void mm_1(cl_mem d_A, cl_mem d_B, cl_mem d_C, int wData, int hData)
{
	uint range_x = wData;
	uint range_y = hData;
	uint group_x = BLOCK_SIZE;
	uint group_y = BLOCK_SIZE;
	
	uint kernel_id = 0;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_A);
	_clSetArgs(kernel_id, arg_idx++, d_B);
	_clSetArgs(kernel_id, arg_idx++, d_C);
	_clSetArgs(kernel_id, arg_idx++, NULL, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, NULL, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &wData, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &hData, sizeof(int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void mm_2(cl_mem d_A, cl_mem d_B, cl_mem d_C, int wData, int hData)
{
	uint range_x = wData;
	uint range_y = hData;
	uint group_x = BLOCK_SIZE;
	uint group_y = BLOCK_SIZE;
	
	uint kernel_id = 1;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_A);
	_clSetArgs(kernel_id, arg_idx++, d_B);
	_clSetArgs(kernel_id, arg_idx++, d_C);
	_clSetArgs(kernel_id, arg_idx++, NULL, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, NULL, BLOCK_SIZE * BLOCK_SIZE * sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &wData, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &hData, sizeof(int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
