#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CLHelper.h"
#include "util.h"

#ifndef TIME
//#define TIME
#endif

#ifndef OUTPUT
//#define OUTPUT
#endif
#define BLOCK_SIZE 512


typedef unsigned int uint;

void reduction_native(cl_mem d_in, cl_mem d_out, uint rake_size, uint N);
void reduction_N(cl_mem d_in, cl_mem d_out, uint rake_size, uint N);
uint OMPReduction(uint * uiIn, uint size);
uint OCLReduction(cl_mem d_in, cl_mem d_out, uint * h_out, uint size);

int main(int argc, char ** argv)
{
	uint * h_in = NULL, * h_out = NULL;
	cl_mem d_in = NULL, d_out = NULL;
try{
	if(argc!=3){
		printf("need 2 parameter here!!!\n");
		exit(-1);
	}
	
	
#if defined TIME
	double start_time = 0;
	double end_time = 0;
	string dat_name="data.dat";

	FILE * fp = fopen(dat_name.c_str(), "a+");
	if(fp==NULL)
	{
		printf("failed to open file!!!\n");
		exit(-1);
	}
#endif

	uint size = atoi(argv[1]); /* array size */
	uint N = atoi(argv[2]); /* granularity */
	uint iter = 1;
	uint s = 32;
	uint blocks = size/(BLOCK_SIZE*2);

	
	printf("size=%d", size);
	h_in = (uint *)malloc(size * sizeof(uint));	
	h_out = (uint *)malloc(blocks * sizeof(uint));
	fill<uint>(h_in, size, 10);
	for(int i=0; i<iter; i++)
	{
		uint uiOmpRes = OMPReduction(h_in, size);
		printf("reduction results: %d\n", uiOmpRes);
	}
	
	//throw(string("manual interuption\n"));
	
	
	_clInit(1, "gpu", 0);
	
	d_in = _clMalloc(size * sizeof(uint));
	d_out = _clMalloc(blocks * sizeof(uint));	
	
	_clMemcpyH2D(d_in, h_in, size * sizeof(uint));

	/**************************1****************************/

	//for(uint g=s; g<N; g=g*2)
	//{
#ifdef TIME
	start_time = gettime();
#endif	
		for(uint i=0; i<iter; i++)
		{
			uint uiOclRes = OCLReduction(d_in, d_out, h_out, size);
			printf("reduction results: %d\n", uiOclRes);
		}
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time-start_time)/(double)iter);	
#endif		
	//}
	
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
	if(h_in!=NULL) free(h_in);
	if(h_out!=NULL) free(h_out);


	return 1;
}
/*
	sequential reduction -- omp vers.	
	ret: the reduction val
*/
uint OMPReduction(uint * uiIn, uint size)
{
	uint res = 0;
	
	for(int i=0; i<size; i++)
	{
		res += uiIn[i];
	}
	return res;
}

/*
	full reduction -- native implementation
*/
void reduction_native(cl_mem d_in, cl_mem d_out, uint rake_size, uint N){

	uint range_x = rake_size;
	uint range_y = 1;
	uint group_x = BLOCK_SIZE;
	uint group_y = 1;
	uint length_x = group_x;
	uint length_y = group_y;
	
	uint kernel_id = 0;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_in);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, &rake_size, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &N, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}


/*
	full reduction -- with local memory
*/
void reduction_N(cl_mem d_in, cl_mem d_out, uint rake_size, uint N){

	uint range_x = rake_size;
	uint range_y = 1;
	uint group_x = BLOCK_SIZE;
	uint group_y = 1;
	uint length_x = group_x;
	uint length_y = group_y;
	
	uint kernel_id = 1;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_in);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, NULL, length_x * length_y * sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &rake_size, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &N, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

/*
	full reduction -- native
*/
uint OCLReduction(cl_mem d_in, cl_mem d_out, uint * h_out, uint size)
{
	uint range_x = size/2;
	uint range_y = 1;
	uint group_x = BLOCK_SIZE;
	uint group_y = 1;
	uint blocks = size/(BLOCK_SIZE*2);
	
	uint kernel_id = 2;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_in);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, &size, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	_clMemcpyD2H(h_out, d_out, blocks * sizeof(uint));
	
	uint res = 0;
	for(int i=0; i<blocks; i++)
	{
		res += h_out[i];
	}
	
	return res;
}


