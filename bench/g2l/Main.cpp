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

#ifndef VARI
//#define VARI
#endif

typedef unsigned int uint;

void g2l_TBT(cl_mem d_in, cl_mem d_out, uint radius, uint w, uint h);
void g2l_FCTH(cl_mem d_in, cl_mem d_out, uint radius, uint w, uint h);
void g2l_CPU(uint * in, uint * out, uint w, uint h, uint r);

int main(int argc, char ** argv)
{
	uint * h_in = NULL, * h_out_1 = NULL, * h_out_2 = NULL, * h_out = NULL;
	cl_mem d_in = NULL, d_out_1 = NULL, d_out_2 = NULL;
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

	uint w, h;
	uint side = atoi(argv[1]);
	w = side, h = side;
	uint size = w * h;
	uint radius = atoi(argv[2]);
	uint iter = 100;

	
	printf("w=%d, h=%d, radius=%d\n", w, h, radius);
	
	_clInit(1, "gpu", 0);

	h_in = (uint *)malloc(size * sizeof(uint));	
	h_out_1 = (uint *)malloc(size * sizeof(uint));
	h_out_2 = (uint *)malloc(size * sizeof(uint));
	h_out = (uint *)malloc(size * sizeof(uint));
	
	d_in = _clMalloc(size * sizeof(uint));
	d_out_1 = _clMalloc(size * sizeof(uint));
	d_out_2 = _clMalloc(size * sizeof(uint));
	
	fill<uint>(h_in, size, 10);
	_clMemcpyH2D(d_in, h_in, size * sizeof(uint));

	//g2l_CPU(h_in, h_out, w, h, radius);

	/**************************1****************************/

#ifdef TIME
	start_time = gettime();
#endif

	for(int i=1; i<iter; i++)
	{
		g2l_TBT(d_in, d_out_1, radius, w, h);
	}
	
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time-start_time)/(double)iter);	
#endif
	
#ifdef VARI
	_clMemcpyD2H(h_out_1, d_out_1, size * sizeof(uint));
#endif	
	
	
	/**************************2****************************/
#ifdef TIME
	start_time = gettime();
#endif
	for(int i=1; i<iter; i++)
	{
		g2l_FCTH(d_in, d_out_2, radius, w, h);
	}
	
#ifdef TIME	
	end_time = gettime();
	fprintf(fp, "%lf\t", (end_time-start_time)/(double)iter);	
#endif

#ifdef VARI	
	_clMemcpyD2H(h_out_2, d_out_2, size * sizeof(uint));
	verify_array_int<uint>(h_out_1, h_out_2, w, h);
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
	_clFree(d_out_1);
	_clFree(d_out_2);
	_clRelease();
	if(h_in!=NULL) free(h_in);
	if(h_out_1!=NULL) free(h_out_1);
	if(h_out_2!=NULL) free(h_out_2);
	if(h_out!=NULL) free(h_out);

	return 1;
}


/*
	in the tile-by-tile mode (TBT)
*/
void g2l_TBT(cl_mem d_in, cl_mem d_out, uint radius, uint w, uint h){

	uint range_x = w;
	uint range_y = h;
	uint group_x = 16;
	uint group_y = 16;
	uint length_x = group_x + 2 * radius;
	uint length_y = group_y + 2 * radius;
	
	uint kernel_id = 0;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_in);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, NULL, length_x * length_y * sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &w, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &h, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &radius, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &length_x, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &length_y, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

/*
	in the First-Central-Then-Halo mode (FCTH)
*/
void g2l_FCTH(cl_mem d_in, cl_mem d_out, uint radius, uint w, uint h){

	uint range_x = w;
	uint range_y = h;
	uint group_x = 16;
	uint group_y = 16;
	uint length_x = group_x + 2 * radius;
	uint length_y = group_y + 2 * radius;
	
	uint kernel_id = 1;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_in);
	_clSetArgs(kernel_id, arg_idx++, d_out);
	_clSetArgs(kernel_id, arg_idx++, NULL, length_x * length_y * sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &w, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &h, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &radius, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &length_x, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &length_y, sizeof(uint));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void g2l_CPU(uint * in, uint * out, uint w, uint h, uint r)
{
	for(uint y=0; y<h; y++)
	{
		for(uint x=0; x<w; x++)
		{
			uint out_idx = y * w + x;
			int in_x , in_y;
			in_x = x - r, in_y = y - r;
			//printf("in_x=%d, in_y=%d\n", in_x, in_y);
			if(in_x < 0) in_x = 0;
			if(in_y < 0) in_y = 0;
			uint in_idx = in_y * w + in_x;
			//printf("in_x=%d, in_y=%d, in_idx=%d\n", in_x, in_y, in_idx);
			uint val = 0;
			val = in[in_idx];
			out[out_idx] = val;
		}
	}
	return ;
}
