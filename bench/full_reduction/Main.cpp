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


typedef unsigned int uint;

void reduction_N(cl_mem d_in, cl_mem d_out, uint rake_size, uint N);

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

	uint size = atoi(argv[1]);
	uint N = atoi(argv[2]);
	uint iter = 1000;

	
	printf("size=%d", size);
	
	_clInit(1, "gpu", 0);

	h_in = (uint *)malloc(size * sizeof(uint));	
	h_out = (uint *)malloc(size * sizeof(uint));
	
	d_in = _clMalloc(size * sizeof(uint));
	d_out = _clMalloc(size * sizeof(uint));
	
	fill<uint>(h_in, size, 10);
	_clMemcpyH2D(d_in, h_in, size * sizeof(uint));


	/**************************1****************************/

	for(uint g=1; g<N; g=g*2)
	{
#ifdef TIME
	start_time = gettime();
#endif
		uint rake_size = size/g;
		for(uint i=1; i<iter; i++)
		{
			reduction_N(d_in, d_out, rake_size, g);
		}
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time-start_time)/(double)iter);	
#endif		
	}
	
	
#ifdef VARI
	_clMemcpyD2H(h_out, d_out, size * sizeof(uint));
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

	_clFree(d_in);
	_clFree(d_out);
	_clRelease();
	if(h_in!=NULL) free(h_in);
	if(h_out!=NULL) free(h_out);


	return 1;
}


/*

*/
void reduction_N(cl_mem d_in, cl_mem d_out, uint rake_size, uint N){

	uint range_x = rake_size;
	uint range_y = 1;
	uint group_x = 512;
	uint group_y = 1;
	uint length_x = group_x;
	uint length_y = group_y;
	
	uint kernel_id = 0;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_in);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, NULL, length_x * length_y * sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &rake_size, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &N, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}


