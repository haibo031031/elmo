#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CLHelper.h"
#include "util.h"
#include <string.h>
#include "shrUtils.h"
#include "oclUtils.h"
#ifndef TIME
#define TIME
#endif

#ifndef VARIFY
//#define VARIFY
#endif

typedef unsigned int uint;
void GPURun(cl_mem devBuffIn, cl_mem devBuffOut, cl_mem devBuffTemp, unsigned int uiImageWidth, unsigned int uiImageHeight, int r);
void GPURunOpt(cl_mem devBuffIn, cl_mem devBuffOut, cl_mem devBuffTemp, unsigned int uiImageWidth, unsigned int uiImageHeight, int r);
void CPURun(uint * in, uint * out, uint wData, uint hData);

static inline size_t DivUp(size_t dividend, size_t divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}


int main(int argc, char ** argv)
{

	unsigned int * uiInput = NULL, * uiTemp = NULL, * uiOutputGPU = NULL, * uiOutputCPU = NULL;
	cl_mem devBufIn = NULL, devBufOut = NULL, devBufTemp = NULL;
	unsigned int uiImageWidth, uiImageHeight, szBufferByte;
	int r;
try{
	if(argc!=2){
		printf("need 1 parameter here!!!");
		exit(-1);
	}
	/* parameters */
	const char * fileName = "lenaRGB.ppm";
	r = 10;
	/* read images */
	shrLoadPPM4ub(fileName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
	//shrSavePPM4ub("output.ppm", (unsigned char *)uiInput, uiImageWidth, uiImageHeight);

	/* allocate memory space on the host */
	szBufferByte = uiImageWidth * uiImageHeight * sizeof(unsigned int);
	uiTemp = (unsigned int *)malloc(szBufferByte);
	uiOutputGPU = (unsigned int *)malloc(szBufferByte);
	uiOutputCPU = (unsigned int *)malloc(szBufferByte);
	
	/* device initialization and context setup */
	_clInit(1, "gpu", 0);
	
	/* create buffer on the device */
	devBufIn = _clMalloc(szBufferByte);
	devBufOut = _clMalloc(szBufferByte);
	devBufTemp = _clMalloc(szBufferByte);
	
	/* copy data from host to device */
	_clMemcpyH2D(devBufIn, uiInput, szBufferByte);
	
	uint iter = 100;
	
#if defined TIME
	double start_time = 0.0;
	double end_time = 0.0;
	double deltaT = 0.0;
	string dat_name="data.dat";

	FILE * fp = fopen(dat_name.c_str(), "a+");
	if(fp==NULL)
	{
		printf("failed to open file!!!\n");
		exit(-1);
	}
#endif
	
	/* warm-up */
	GPURun(devBufIn, devBufOut, devBufTemp, uiImageWidth, uiImageHeight, r);
	GPURunOpt(devBufIn, devBufOut, devBufTemp, uiImageWidth, uiImageHeight, r);
	_clMemcpyD2H(uiOutputGPU, devBufTemp, szBufferByte);
	shrSavePPM4ub("output.ppm", (unsigned char *)uiOutputGPU, uiImageWidth, uiImageHeight);
#ifdef VARIFY	
	CPURun(in, out_cpu, wData, hData);
#endif //VARIFY
	
	/**************************1****************************/
#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		GPURun(devBufIn, devBufOut, devBufTemp, uiImageWidth, uiImageHeight, r);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

	/**************************1****************************/
#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		GPURunOpt(devBufIn, devBufOut, devBufTemp, uiImageWidth, uiImageHeight, r);
	
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

	_clFree(devBufIn);
	_clFree(devBufOut);
	_clFree(devBufTemp);
	_clRelease();
	if(uiInput!=NULL) free(uiInput);
	if(uiTemp!=NULL) free(uiTemp);
	if(uiOutputGPU!=NULL) free(uiOutputGPU);
	if(uiOutputCPU!=NULL) free(uiOutputCPU);

	return 1;
}

/*
	matrix tranpose on CPU as reference
*/
void CPURun(uint * in, uint * out, uint wData, uint hData)
{
	for(int y=0; y<hData; y++)
	{
		for(int x=0; x<wData; x++)
		{
			uint val = in[x * wData + y];
			out[y * hData + x] = val;
		}
	}
	
	return ;
}

/*
	Reference impl. of BoxFilter on GPUs
*/
void GPURun(cl_mem devBuffIn, cl_mem devBuffOut, cl_mem devBuffTemp, unsigned int uiImageWidth, unsigned int uiImageHeight, int r)
{
	/* row transformation */
	uint uiRadiusAligned = ((r+15)/16)*16;
	uint uiNumOutputPix = 64;
	float fScale = 1.0f/(2.0f * r + 1.0f);
	uint group_x = uiNumOutputPix + uiRadiusAligned + r;
	uint group_y = 1;
	uint range_x = group_x * DivUp(uiImageWidth, uiNumOutputPix);
	uint range_y = uiImageHeight;
	
	printf("--work-group size:%d\n", group_x);
	uint kernel_id = 0;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, devBuffIn);
	_clSetArgs(kernel_id, arg_idx++, devBuffTemp);
	_clSetArgs(kernel_id, arg_idx++, NULL, group_x * sizeof(cl_uchar4));
	_clSetArgs(kernel_id, arg_idx++, &uiImageWidth, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &uiImageHeight, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &r, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &uiRadiusAligned, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &fScale, sizeof(float));
	_clSetArgs(kernel_id, arg_idx++, &uiNumOutputPix, sizeof(unsigned int));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	/* column transformation */
	range_x = uiImageWidth;
	range_y = 1;
	group_x = 64;
	group_y = 1;
	
	kernel_id = 1;
	arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, devBuffTemp);
	_clSetArgs(kernel_id, arg_idx++, devBuffOut);
	_clSetArgs(kernel_id, arg_idx++, &uiImageWidth, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &uiImageHeight, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &r, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &fScale, sizeof(float));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);	
	
	return ;
}

/*
	Optimized impl. of BoxFilter on GPUs
*/
void GPURunOpt(cl_mem devBuffIn, cl_mem devBuffOut, cl_mem devBuffTemp, unsigned int uiImageWidth, unsigned int uiImageHeight, int r)
{
	/* row transformation */
	uint uiRadiusAligned = ((r+15)/16)*16;
	uint uiNumOutputPix = 64;
	float fScale = 1.0f/(2.0f * r + 1.0f);
	//uint group_x = uiNumOutputPix + uiRadiusAligned + r;
	uint group_x = uiNumOutputPix;
	uint group_y = 1;
	//uint range_x = group_x * DivUp(uiImageWidth, uiNumOutputPix);
	uint range_x = uiImageWidth;
	uint range_y = uiImageHeight;
	
	printf("--work-group size:%d\n", group_x);
	uint kernel_id = 2;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, devBuffIn);
	_clSetArgs(kernel_id, arg_idx++, devBuffTemp);
	//_clSetArgs(kernel_id, arg_idx++, NULL, group_x * sizeof(cl_uchar4));
	_clSetArgs(kernel_id, arg_idx++, NULL, (group_x + 2 * r) * group_y * sizeof(cl_uchar4));
	_clSetArgs(kernel_id, arg_idx++, &uiImageWidth, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &uiImageHeight, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &r, sizeof(int));
	//_clSetArgs(kernel_id, arg_idx++, &uiRadiusAligned, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &fScale, sizeof(float));
	//_clSetArgs(kernel_id, arg_idx++, &uiNumOutputPix, sizeof(unsigned int));

	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	/* column transformation */
	range_x = uiImageWidth;
	range_y = 1;
	group_x = 64;
	group_y = 1;
	
	kernel_id = 1;
	arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, devBuffTemp);
	_clSetArgs(kernel_id, arg_idx++, devBuffOut);
	_clSetArgs(kernel_id, arg_idx++, &uiImageWidth, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &uiImageHeight, sizeof(unsigned int));
	_clSetArgs(kernel_id, arg_idx++, &r, sizeof(int));
	_clSetArgs(kernel_id, arg_idx++, &fScale, sizeof(float));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

