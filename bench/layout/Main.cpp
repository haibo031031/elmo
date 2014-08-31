#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CLHelper.h"
#include "util.h"

#ifndef TIME
#define TIME
#endif

#ifndef OUTPUT
//#define OUTPUT
#endif

#define BS 64

typedef unsigned int uint;

void layout_blocked(cl_mem out, uint bins, uint size);
void layout_cyclic(cl_mem out, uint bins, uint size);
void layout_cyclic_2(cl_mem out, uint bins, uint size);

int main(int argc, char ** argv)
{
	cl_mem out = NULL;
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

	uint bins = atoi(argv[1]);
	uint size = atoi(argv[2]);
	uint iter = 100;

	
	printf("bins=%d, size=%d\n", bins, size);
	
	_clInit(1, "gpu", 0);
	
	out = _clMalloc((size/BS)*bins);
	
	layout_cyclic(out, bins, size);
	
	/**************************1****************************/

#ifdef TIME
	start_time = gettime();
#endif

	for(int i=1; i<iter; i++)
	{
		layout_blocked(out, bins, size);
	}
	
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time-start_time)/(double)iter);	
#endif		

	/**************************2****************************/

#ifdef TIME
	start_time = gettime();
#endif

	for(int i=1; i<iter; i++)
	{
		layout_cyclic(out, bins, size);
	}
	
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time-start_time)/(double)iter);	
#endif

	/**************************3****************************/

#ifdef TIME
	start_time = gettime();
#endif

	for(int i=1; i<iter; i++)
	{
		layout_cyclic_2(out, bins, size);
	}
	
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time-start_time)/(double)iter);	
#endif


#ifdef TIME	
	fprintf(fp, "\n");	
	fclose(fp);
#endif	
}
catch(string msg){
	printf("ERR:%s\n", msg.c_str());
	printf("Error catched\n");
	}

	_clFree(out);
	_clRelease();
	return 1;
}


/*
	organize the local memory in the blocked mode [blocked]
*/
void layout_blocked(cl_mem out, uint bins, uint size){

	uint range_x = size;
	uint range_y = 1;
	uint group_x = BS;
	uint group_y = 1;
	
	uint sz_lm = group_x * bins;
		
	uint kernel_id = 0;
	uint arg_idx = 0;	
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, sz_lm * sizeof(unsigned char));
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

/*

	organize the local memory in the cyclic mode [cyclic]
*/
void layout_cyclic(cl_mem out, uint bins, uint size){

	uint range_x = size;
	uint range_y = 1;
	uint group_x = BS;
	uint group_y = 1;
	
	uint sz_lm = group_x * bins;
		
	uint kernel_id = 1;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, sz_lm * sizeof(unsigned char));
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void layout_cyclic_2(cl_mem out, uint bins, uint size){

	uint range_x = size;
	uint range_y = 1;
	uint group_x = BS;
	uint group_y = 1;
	
	uint sz_lm = group_x * bins;
		
	uint kernel_id = 2;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, sz_lm * sizeof(unsigned char));
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

