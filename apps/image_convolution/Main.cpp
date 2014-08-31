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

void CPURun(uint * in, uint * filter, uint * out, uint wData, uint hData, int r);

void convolve_1(cl_mem in, cl_mem filter, cl_mem out, uint wData, uint hData, int r);
void convolve_2(cl_mem in, cl_mem filter, cl_mem out, uint wData, uint hData, int r);
void convolve_3(cl_mem in, cl_mem filter, cl_mem out, uint wData, uint hData, int r);

int main(int argc, char ** argv)
{
	uint * in = NULL, * out_cpu = NULL, * out_gpu = NULL, * filter = NULL;
	cl_mem d_in = NULL, d_out = NULL, d_filter = NULL;
try{
	if(argc!=3){
		printf("need 2 parameter here!!!");
		exit(-1);
	}

	_clInit(0, "cpu", 0);
	uint iter = 100;
	
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
	
	// parameters
	uint side = atoi(argv[1]);
	uint wData = side;
	uint hData = side;
	uint size = wData * hData;
	uint r = atoi(argv[2]);
	uint area = (2 * r + 1) * (2 * r + 1);

	printf("wData=%d, hData=%d, r=%d\n", wData, hData, r);
	
	// allocate memory space on the host and device side
	in = (uint * )malloc(size * sizeof(uint));
	out_cpu = (uint * )malloc(size * sizeof(uint));
	out_gpu = (uint * )malloc(size * sizeof(uint));
	filter = (uint * )malloc(area * sizeof(uint));
	
	d_in = _clMalloc(size * sizeof(uint));	
	d_out = _clMalloc(size * sizeof(uint));
	d_filter = _clMalloc(area * sizeof(uint));

	// initialization
	fill<uint>(in, size, 16);
	fill<uint>(filter, area, 16);

	// copy data from host to device
	_clMemcpyH2D(d_in, in, size * sizeof(uint));
	_clMemcpyH2D(d_filter, filter, area * sizeof(uint));
	
	// warm-up
	convolve_1(d_in, d_filter, d_out, wData, hData, r);
	convolve_2(d_in, d_filter, d_out, wData, hData, r);
	convolve_3(d_in, d_filter, d_out, wData, hData, r);
	
#ifdef VARIFY	
	CPURun(in, filter, out_cpu, wData, hData, r);
#endif //VARIFY
	
	/**************************1****************************/
#ifdef TIME
	double deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		convolve_1(d_in, d_filter, d_out, wData, hData, r);
	
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

		convolve_2(d_in, d_filter, d_out, wData, hData, r);
	
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

		convolve_3(d_in, d_filter, d_out, wData, hData, r);
	
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
	_clFree(d_filter);
	_clRelease();
	if(in!=NULL) free(in);
	if(out_cpu!=NULL) free(out_cpu);
	if(out_gpu!=NULL) free(out_gpu);
	if(filter!=NULL) free(filter);

	return 1;
}

void CPURun(uint * in, uint * filter, uint * out, uint wData, uint hData, int r)
{
	for(int y=0; y<hData; y++)
	{
		for(int x=0; x<wData; x++)
		{
			uint val = 0;	
			uint wF = 2 * r + 1;
		    for(int _y=-r, __y=0; _y<=r; _y++, __y++)
    		{
		    	for(int _x=-r, __x=0; _x<=r; _x++, __x++)
		    	{
    				int d_gl_x = x + _x;
    				if(d_gl_x<0) d_gl_x = 0;
    				else if(d_gl_x>=wData) d_gl_x = wData - 1;    		
    				int d_gl_y = y + _y;
    				if(d_gl_y<0) d_gl_y = 0;
    				else if(d_gl_y>=hData) d_gl_y = hData - 1;    		
    				int d_gl_in = d_gl_y * wData + d_gl_x;    		
    				int d_f_idx = __y * wF + __x;
    				
    				val = val + in[d_gl_in] * filter[d_f_idx];
    				//printf("(%d, %d)\t", in[d_gl_in] * filter[d_f_idx], val);
    			}
    		}
    		//printf("(%d, %d)\t", in[y * wData + x], val);
    		val = val/((2 * r + 1)*(2 * r + 1));
		    uint d_gl_idx = y * wData + x;
		    out[d_gl_idx] = val;
		}
	}
	
	return ;
}

void convolve_1(cl_mem in, cl_mem filter, cl_mem out, uint wData, uint hData, int r)
{
	uint range_x = wData;
	uint range_y = hData;
	uint group_x = 16;
	uint group_y = 16;
	
	uint kernel_id = 0;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, in);
	_clSetArgs(kernel_id, arg_idx++, filter);
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, &wData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &hData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &r, sizeof(int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void convolve_2(cl_mem in, cl_mem filter, cl_mem out, uint wData, uint hData, int r)
{
	uint range_x = wData;
	uint range_y = hData;
	uint group_x = 16;
	uint group_y = 16;
	uint length_x = group_x + (2 * r);
	uint length_y = group_y + (2 * r);
	
	uint kernel_id = 1;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, in);
	_clSetArgs(kernel_id, arg_idx++, filter);
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, length_x * length_y * sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &wData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &hData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &r, sizeof(int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void convolve_3(cl_mem in, cl_mem filter, cl_mem out, uint wData, uint hData, int r)
{
	uint range_x = wData;
	uint range_y = hData;
	uint group_x = 16;
	uint group_y = 16;
	uint length_x = group_x + (2 * r);
	uint length_y = group_y + (2 * r);
	
	uint kernel_id = 2;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, in);
	_clSetArgs(kernel_id, arg_idx++, filter);
	_clSetArgs(kernel_id, arg_idx++, out);
	_clSetArgs(kernel_id, arg_idx++, NULL, length_x * length_y * sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &wData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &hData, sizeof(uint));
	_clSetArgs(kernel_id, arg_idx++, &r, sizeof(int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

