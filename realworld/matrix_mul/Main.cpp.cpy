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
	float * A = NULL, * B = NULL, * C = NULL, *refC;
	cl_mem d_A = NULL, d_B = NULL, d_C = NULL;
try{
	/*if(argc!=2){
		printf("need 1 parameter here!!!");
		exit(-1);
	}*/

	_clInit(0, "cpu", 0);
	uint iter = 100;
	
#if defined TIME
	double start_time = 0.0;
	double end_time = 0.0;
	double deltaT = 0.0;
	string dat_name="data.dat";

	FILE * fp = fopen(dat_name.c_str(), "a+");
	if(fp==NULL)
	{
		printf("failed to open file!!!\n");
		exit(-1);
	}
#endif
	
	// parameters
	//uint side = atoi(argv[1]);
	uint side = 128;
	uint wData = side;
	uint hData = side;
	uint size = wData * hData;

	printf("wData=%d, hData=%d\n", wData, hData);
	
	// allocate memory space on the host and device side
	/*A = (float * )malloc(size * sizeof(float));
	B = (float * )malloc(size * sizeof(float));
	C = (float * )malloc(size * sizeof(float));
	refC = (float * )malloc(size * sizeof(float));
	
	d_A = _clMalloc(size * sizeof(cl_float));	
	d_B = _clMalloc(size * sizeof(cl_float));	
	d_C = _clMalloc(size * sizeof(cl_float));

	// initialization
	fill<float>(A, size, 16);
	fill<float>(B, size, 16);

	// copy data from host to device
	_clMemcpyH2D(d_A, A, size * sizeof(float));
	_clMemcpyH2D(d_B, B, size * sizeof(float));
	
	// warm-up
	mm_1(d_A, d_B, d_C, wData, hData);
	mm_2(d_A, d_B, d_C, wData, hData);
	
#ifdef VARIFY	
	CPURun(A, B, refC, wData, hData);
#endif //VARIFY*/
	
	/**************************1****************************/
/*#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		mm_1(d_A, d_B, d_C, wData, hData);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(C, d_C, size * sizeof(float));
	verify_array<float>(C, refC, size);
#endif //VARIFY*/

	/**************************2****************************/
/*#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		mm_2(d_A, d_B, d_C, wData, hData);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(C, d_C, size * sizeof(float));
	verify_array<float>(C, refC, size);
#endif //VARIFY

#ifdef TIME	
	fprintf(fp, "\n");	
	fclose(fp);
#endif	
}
	catch(string msg)
	{
		printf("ERR:%s\n", msg.c_str());
	}

	_clFree(d_A);
	_clFree(d_B);
	_clFree(d_C);
	_clRelease();
	if(A!=NULL) free(A);
	if(B!=NULL) free(B);
	if(C!=NULL) free(C);
	if(refC!=NULL) free(refC);*/

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
