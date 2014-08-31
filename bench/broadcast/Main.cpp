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

void CPURun(const float * in, float * out, int cdim, int rdim);
void broadcast(cl_mem raw_cost, cl_mem d_out, int cdim, int rdim);
void broadcast_lm(cl_mem raw_cost, cl_mem d_out, int cdim, int rdim);

int main(int argc, char ** argv)
{
	float * h_raw, * h_out, * outCPU;
	cl_mem d_raw, d_out;
try{
	if(argc!=2){
		printf("need one parameter here!!!");
		exit(-1);
	}

	_clInit(1, "gpu", 0);

	
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

	int cdim = atoi(argv[1]); //{384};
	int rdim = atoi(argv[1]); //{288};
	printf("cdim=%d, rdim=%d\n", cdim, rdim);
	h_raw = (float *)malloc(cdim * rdim * sizeof(float));
	h_out = (float *)malloc(cdim * rdim * sizeof(float));
	outCPU = (float *)malloc(cdim * rdim * sizeof(float));
	fill<float>(h_raw, cdim * rdim, 5);	
	d_raw = _clMalloc(cdim * rdim * sizeof(float));
	d_out = _clMalloc(cdim * rdim * sizeof(float));
	_clMemcpyH2D(d_raw, h_raw, cdim * rdim * sizeof(float));
	printf("-0\n");
#ifdef VARIFY	
	CPURun(h_raw, outCPU, cdim, rdim);
#endif //VARIFY
	
	/**************************1****************************/
#ifdef TIME
	start_time = gettime();
#endif
	printf("-1.1\n");
	broadcast(d_raw, d_out, cdim, rdim);
	printf("-1.2\n");
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time));
#endif	

#ifdef VARIFY
	_clMemcpyD2H(h_out, d_out, cdim * rdim * sizeof(float));	
	verify_array<float>(outCPU, h_out, cdim * rdim);	
#endif //VARIFY

	/**************************2****************************/
#ifdef TIME
	start_time = gettime();	
#endif
	broadcast_lm(d_raw, d_out, cdim, rdim);
	printf("-2\n");
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time));	
#endif	

#ifdef VARIFY
	_clMemcpyD2H(h_out, d_out, cdim * rdim * sizeof(float));
	verify_array<float>(outCPU, h_out, cdim * rdim);
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
	_clRelease();
	if(h_raw!=NULL) free(h_raw);
	if(h_out!=NULL) free(h_out);
	if(outCPU!=NULL) free(outCPU);

	return 1;
}

void CPURun(const float * in, float * out, int cdim, int rdim)
{
	//int x = get_global_id(0);
	//int y = get_global_id(1);
	//float curCost = 0.0f, deltaCost = 0.0f;
	//if(x<cdim && y<rdim)
	//{
#pragma omp parallel for 	
	for(int y=0; y<rdim; y++)
	{
		for(int x=0; x<cdim; x++)
		{
			int idx = y * cdim + x;
			float curCost = in[idx];
			float deltaCost = 0.0f;
			for(int _y=0; _y<rdim; _y++)
			{
				for(int _x=0; _x<cdim; _x++)
				{
					float tCost = in[_y * cdim + _x];
					deltaCost += curCost - tCost;
				}
			}
			out[idx] = deltaCost;			
		}
	}

	//}
}

void broadcast(cl_mem raw_cost, cl_mem d_out, int cdim, int rdim){
	int kernel_id = 0;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, raw_cost);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void broadcast_lm(cl_mem raw_cost, cl_mem d_out, int cdim, int rdim){

	int range_x = cdim;
	int range_y = rdim;
	int group_x = 16;
	int group_y = 16;

	int kernel_id = 1;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, raw_cost);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, NULL, group_x * group_y * sizeof(float));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);	
	
	return ;
}
