#include <stdlib.h>
#include <stdio.h>
#include "CLHelper.h"
#include "util.h"

#ifndef TIME
#define TIME
#endif

#define BOX 16

void reg_pres(cl_mem raw_cost, int rdim, int cdim);
void reg_pres_lm(cl_mem raw_cost, int rdim, int cdim);
void reg_pres_gm(cl_mem raw_cost, int rdim, int cdim, cl_mem hist);

int main(int argc, char ** argv)
{
try{
	if(argc!=2){
		printf("need one parameter here!!!");
		exit(-1);
	}

	_clInit(1, "gpu", 0);

	
#if defined TIME
	double start_time = 0;
	double end_time = 0;
	string dat_name="reg_pres.dat";

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
	int * h_raw = (int *)malloc(cdim * rdim * sizeof(int));
	fill<int>(h_raw, cdim * rdim, BOX);	
	cl_mem d_raw = _clMalloc(cdim * rdim * sizeof(int));
	cl_mem d_hist = _clMalloc(cdim * rdim * BOX * sizeof(int));
	_clMemcpyH2D(d_raw, h_raw, cdim * rdim * sizeof(int));
	
	/**************************1****************************/
#ifdef TIME
	start_time = gettime();	
#endif
	reg_pres(d_raw, rdim, cdim);
	printf("-1\n");
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time));	
#endif	


	fill<int>(h_raw, cdim * rdim, BOX);	
	_clMemcpyH2D(d_raw, h_raw, cdim * rdim * sizeof(int));
	
	
	/**************************2****************************/
#ifdef TIME
	start_time = gettime();	
#endif
	reg_pres_lm(d_raw, rdim, cdim);
	printf("-2\n");
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time));	
#endif	

	/**************************2****************************/
#ifdef TIME
	start_time = gettime();	
#endif
	reg_pres_gm(d_raw, rdim, cdim, d_hist);
	printf("-3\n");
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time - start_time));	
#endif	


#ifdef TIME	
	fprintf(fp, "\n");	
	fclose(fp);
#endif	

	_clFree(d_raw);
	_clRelease();
	if(h_raw!=NULL) free(h_raw);
}
catch(string msg){
	printf("ERR:%s\n", msg.c_str());
	printf("Error catched\n");
}
return 1;
}


void reg_pres(cl_mem raw_cost, int rdim, int cdim){
	int kernel_id = 0;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, raw_cost);
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	int range_x = rdim;
	int range_y = 1;
	int group_x = 64;
	int group_y = 1;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void reg_pres_lm(cl_mem raw_cost, int rdim, int cdim){
	int kernel_id = 1;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, raw_cost);
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, NULL, 64 * BOX * sizeof(int));
	int range_x = rdim;
	int range_y = 1;
	int group_x = 64;
	int group_y = 1;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void reg_pres_gm(cl_mem raw_cost, int rdim, int cdim, cl_mem hist){
	int kernel_id = 2;
	int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, raw_cost);
	_clSetArgs(kernel_id, arg_idx++, &rdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &cdim, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, hist);
	int range_x = rdim;
	int range_y = 1;
	int group_x = 64;
	int group_y = 1;
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
