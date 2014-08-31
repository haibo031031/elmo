#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CLHelper.h"
#include "util.h"

#ifndef TIME
#define TIME
#endif


void init_bc(int size, int wg_size, int bins);
void init_bc_free(int size, int wg_size, int bins);



int main(int argc, char ** argv)
{

try{
	if(argc!=4){
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
	
	int size = atoi(argv[1]);
	int wg_size = atoi(argv[2]);
	int bins = atoi(argv[3]);
	printf("size=%d, work-group size=%d, bins=%d\n", size, wg_size, bins);

	
	/**************************1****************************/
#ifdef TIME
	start_time = gettime();
#endif
	for(int i=0; i<iter; i++)
	{
		init_bc(size, wg_size, bins);
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
		init_bc_free(size, wg_size, bins);
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
	_clRelease();
	return 1;
}


void init_bc(int size, int wg_size, int bins){

	int range_x = size;
	int range_y = 1;
	int group_x = wg_size;
	int group_y = 1;

	int kernel_id = 0;
	int arg_idx = 0;

	_clSetArgs(kernel_id, arg_idx++, NULL, group_x * group_y * bins * sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(int));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void init_bc_free(int size, int wg_size, int bins){

	
	int range_x = size;
	int range_y = 1;
	int group_x = wg_size;
	int group_y = 1;

	int kernel_id = 1;
	int arg_idx = 0;

	_clSetArgs(kernel_id, arg_idx++, NULL, group_x * group_y * bins * sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(int));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
