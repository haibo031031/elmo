#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CLHelper.h"
#include "util.h"

#ifndef TIME
#define TIME
#endif


void native(cl_mem out, int side);
void opt(cl_mem out, int side);



int main(int argc, char ** argv)
{
	cl_mem out = NULL;
try{
	if(argc!=2){
		printf("need one parameter here!!!");
		exit(-1);
	}

	_clInit(1, "gpu", 0);
	int iter = 10;
	
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
	
	int side = atoi(argv[1]);
	int size = side * side;

	printf("side=%d\n", side);
	
	out = _clMalloc(size * sizeof(float));

	// warm up
	native(out, side);
	opt(out, side);
	/**************************1****************************/
#ifdef TIME
	start_time = gettime();
#endif
	for(int i=0; i<iter; i++)
	{
		native(out, side);
	}
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time)/(double)iter);
#endif	


	/**************************2****************************/
#ifdef TIME
	start_time = gettime();	
#endif
	for(int i=0; i<iter; i++)
	{
		opt(out, side);
	}

#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time)/(double)iter);
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


void native(cl_mem out, int side){

	int range_x = side;
	int range_y = side;
	int group_x = 16;
	int group_y = 16;

	int kernel_id = 0;
	int arg_idx = 0;
	
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, group_x * group_y * sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &side, sizeof(int));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}


void opt(cl_mem out, int side){

	int range_x = side;
	int range_y = side;
	int group_x = 16;
	int group_y = 16;

	int kernel_id = 1;
	int arg_idx = 0;
	
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, (group_x+1) * group_y * sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &side, sizeof(int));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
