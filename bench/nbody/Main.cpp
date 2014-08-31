#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CLHelper.h"
#include "util.h"

#ifndef TIME
#define TIME
#endif

#ifndef VARIFY
//#define VARIFY
#endif


#define BOX 16

void CPURun(const float * in, float * out, int size);
void broadcast(cl_mem raw_cost, cl_mem d_out, int size, int wg_size);
void broadcast_lm(cl_mem raw_cost, cl_mem d_out, int size, int wg_size);

int main(int argc, char ** argv)
{
	float * h_raw, * h_out, * outCPU;
	cl_mem d_raw, d_out, d_out_lm;
try{
	if(argc!=3){
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
	printf("size=%d, work-group size=%d\n", size, wg_size);
	h_raw = (float *)malloc(size * sizeof(float));
	h_out = (float *)malloc(size * sizeof(float));
	outCPU = (float *)malloc(size * sizeof(float));
	fill<float>(h_raw, size, 5);	
	d_raw = _clMalloc(size * sizeof(float));
	d_out = _clMalloc(size * sizeof(float));
	d_out_lm = _clMalloc(size * sizeof(float));
	_clMemcpyH2D(d_raw, h_raw, size * sizeof(float));
#ifdef VARIFY	
	CPURun(h_raw, outCPU, size);
#endif //VARIFY
	
	/**************************1****************************/
#ifdef TIME
	start_time = gettime();
#endif
	for(int i=0; i<iter; i++)
	{
		broadcast(d_raw, d_out, size, wg_size);
	}
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time)/(double)iter);
#endif	

#ifdef VARIFY
	_clMemcpyD2H(h_out, d_out, size * sizeof(float));
	verify_array<float>(outCPU, h_out, size);
#endif //VARIFY

	/**************************2****************************/
#ifdef TIME
	start_time = gettime();	
#endif
	for(int i=0; i<iter; i++)
	{
		broadcast_lm(d_raw, d_out_lm, size, wg_size);
	}

#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time)/(double)iter);
#endif	

#ifdef VARIFY
	_clMemcpyD2H(h_out, d_out_lm, size * sizeof(float));
	verify_array<float>(outCPU, h_out, size);
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
	_clFree(d_raw);
	_clFree(d_out);
	_clFree(d_out_lm);
	_clRelease();
	if(h_raw!=NULL) free(h_raw);
	if(h_out!=NULL) free(h_out);
	if(outCPU!=NULL) free(outCPU);

	return 1;
}

void CPURun(const float * in, float * out, int size, int wg_size)
{
#pragma omp parallel for 	
	for(int x=0; x<size; x++)
	{
		float curCost = in[x];
		float deltaCost = 0.0f;
		for(int _x=0; _x<size; _x++)
		{
			float tCost = in[_x];
			deltaCost += curCost - tCost;
		}
		out[x] = deltaCost;			
	}
}

void broadcast(cl_mem raw_cost, cl_mem d_out, int size, int wg_size){

	int range_x = size;
	int range_y = 1;
	int group_x = wg_size;
	int group_y = 1;

	int kernel_id = 0;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, raw_cost);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, &size, sizeof(int));
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void broadcast_lm(cl_mem raw_cost, cl_mem d_out, int size, int wg_size){

	int range_x = size;
	int range_y = 1;
	int group_x = wg_size;
	int group_y = 1;

	int kernel_id = 1;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, raw_cost);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, &size, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, NULL, group_x * group_y * sizeof(float));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);	
	
	return ;
}
