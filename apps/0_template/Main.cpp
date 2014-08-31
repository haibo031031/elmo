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
//#define VARIFY
#endif

typedef unsigned int uint;

void CPURun(uint * in, uint * out, uint wData, uint hData);

void mt_1(cl_mem in, cl_mem out, uint wData, uint hData);
void mt_2(cl_mem in, cl_mem out, uint wData, uint hData);
void mt_3(cl_mem in, cl_mem out, uint wData, uint hData);

int main(int argc, char ** argv)
{
	uint * in = NULL, * out_cpu = NULL, * out_gpu = NULL;
	cl_mem d_in = NULL, d_out = NULL;
try{
	if(argc!=2){
		printf("need 1 parameter here!!!");
		exit(-1);
	}

	_clInit(1, "gpu", 0);
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
	uint side = atoi(argv[1]);
	uint wData = side;
	uint hData = side;
	uint size = wData * hData;

	printf("wData=%d, hData=%d\n", wData, hData);
	
	// allocate memory space on the host and device side
	in = (uint * )malloc(size * sizeof(uint));
	out_cpu = (uint * )malloc(size * sizeof(uint));
	out_gpu = (uint * )malloc(size * sizeof(uint));
	
	d_in = _clMalloc(size * sizeof(uint));	
	d_out = _clMalloc(size * sizeof(uint));

	// initialization
	fill<uint>(in, size, 16);

	// copy data from host to device
	_clMemcpyH2D(d_in, in, size * sizeof(uint));
	
	// warm-up
	mt_1(d_in, d_out, wData, hData);
	mt_2(d_in, d_out, wData, hData);
	mt_3(d_in, d_out, wData, hData);
	
#ifdef VARIFY	
	CPURun(in, out_cpu, wData, hData);
#endif //VARIFY
	
	/**************************1****************************/
#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		mt_1(d_in, d_out, wData, hData);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(out_gpu, d_out, size * sizeof(uint));
	verify_array_int<uint>(out_cpu, out_gpu, size);
#endif //VARIFY

	/**************************2****************************/
#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		mt_2(d_in, d_out, wData, hData);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(out_gpu, d_out, size * sizeof(uint));
	verify_array_int<uint>(out_cpu, out_gpu, size);
#endif //VARIFY

	/**************************3****************************/
#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		mt_3(d_in, d_out, wData, hData);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(out_gpu, d_out, size * sizeof(uint));
	verify_array_int<uint>(out_cpu, out_gpu, size);
#endif //VARIFY

#ifdef TIME	
	fprintf(fp, "\n");	
	fclose(fp);
#endif	
}
catch(string msg){
	printf("ERR:%s\n", msg.c_str());
	printf("Error catched\n");
	}

	_clFree(d_in);
	_clFree(d_out);
	_clRelease();
	if(in!=NULL) free(in);
	if(out_cpu!=NULL) free(out_cpu);
	if(out_gpu!=NULL) free(out_gpu);

	return 1;
}

/*
	matrix tranpose on CPU as reference
*/
void CPURun(uint * in, uint * out, uint wData, uint hData)
{
	for(int y=0; y<hData; y++)
	{
		for(int x=0; x<wData; x++)
		{
			uint val = in[x * wData + y];
			out[y * hData + x] = val;
		}
	}
	
	return ;
}

/*
	Naive impl. of MT on GPUs
*/
void mt_1(cl_mem in, cl_mem out, uint wData, uint hData)
{
	uint range_x = wData;
	uint range_y = hData;
	uint group_x = 16;
	uint group_y = 16;
	
	uint kernel_id = 0;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, in);
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, &wData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &hData, sizeof(uint));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
/*
	opted impl. of MT on GPUs
*/

void mt_2(cl_mem in, cl_mem out, uint wData, uint hData)
{
	uint range_x = wData;
	uint range_y = hData;
	uint group_x = 16;
	uint group_y = 16;
	uint length_x = group_x;
	uint length_y = group_y;
	
	uint kernel_id = 1;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, in);
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, length_x * length_y * sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &wData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &hData, sizeof(uint));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

/*
	opted impl. of MT on GPUs -- bank-conflicts removal
*/

void mt_3(cl_mem in, cl_mem out, uint wData, uint hData)
{
	uint range_x = wData;
	uint range_y = hData;
	uint group_x = 16;
	uint group_y = 16;
	uint length_x = group_x + 1; // bank-conflicts removal
	uint length_y = group_y;
	
	uint kernel_id = 2;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, in);
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, length_x * length_y * sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &wData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &hData, sizeof(uint));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
