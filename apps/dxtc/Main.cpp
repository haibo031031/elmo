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
//void OMPRun8x8(float * dst, float * src, uint W, uint H, int dir);
//void OCLRun8x8(cl_mem devIn, cl_mem devOut, uint uiImageW, uint uiImageH, const int dir);

static inline uint iDivUp(uint dividend, uint divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

const char *image_filename = "lena_std.ppm";
const char *refimage_filename = "lena_ref.dds";
#define ERROR_THRESHOLD 0.02f
#define NUM_THREADS   64      // Number of threads per work group.


int main(int argc, char ** argv)
{
	uint * uiIn = NULL, * uiOutOcl = NULL, * uiOutOmp = NULL;
	uint * uiInB = NULL;
	cl_mem devIn = NULL, devOut = NULL;
	
	
try{
	_clParseCommandLine(argc, argv);
	
	/* parameters */
	uint uiImageW = 8, uiImageH = 8;
	string strSubfix = argv[1];
	uint uiNumElems = 1;
	
	/* load image */
	shrLoadPPM4ub(image_filename, (unsigned char **)&uiIn, &uiImageW, &uiImageH);
	printf("Loaded %s, %d x %d pixels\n\n", image_filename, uiImageW, uiImageH);
	uiNumElems = uiImageW * uiImageH;
	
	/* image conversion: from linear to block */	
    uint uiMemSize = uiNumElems * sizeof(uint);
    uiInB = (uint * )malloc(memSize);

    for(uint by = 0; by < uiImageH/4; by++) {
        for(uint bx = 0; bx < uiImageW/4; bx++) {
            for (int i = 0; i < 16; i++) {
                const int x = i & 3;
                const int y = i / 4;
                uiInB[(by * uiImageW/4 + bx) * 16 + i] = 
                    ((uint *)uiIn)[(by * 4 + y) * 4 * (uiImageW/4) + bx * 4 + x];
            }
        }
    }

	
	/* allocate memory space on the host */

	/* initialization */

	const uint iter = 10;
	//OMPRun8x8(fOutOmp, fIn, uiImageW, uiImageH, 0);

	
#if defined TIME
	double start_time = 0.0;
	double end_time = 0.0;
	double deltaT = 0.0;
	string dat_name= string("data_") + strSubfix + string(".dat");

	FILE * fp = fopen(dat_name.c_str(), "a+");
	if(fp==NULL)
	{
		printf("failed to open file!!!\n");
		exit(-1);
	}
#endif
	
#if defined VARIFY
	/* ---------------------
		omp version on cpu 
		---------------------*/	
	printf("--------------------------\n");
#ifdef TIME
	deltaT = 0.0;
#endif	
	for(int i=0; i<iter; i++){
#ifdef TIME	
	start_time = gettime();
#endif	
	//OMPRun8x8(fOutOmp, fIn, uiImageW, uiImageH, dir);
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;		
#endif	
	}
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif
	
	printf("omp version on cpu\n\n");
#endif

	/* ---------------------
		ocl version on xpu
		---------------------*/	
	printf("--------------------------\n");		
	/* device initialization and context setup */
	_clInit(platform_id, device_type, device_id);	

	/* create buffer on the device */
	//devIn = _clMalloc(szBufferByte);
	//devOut = _clMalloc(szBufferByte);
#ifdef TIME
	deltaT = 0.0;
#endif		
	
	for(int i=0; i<iter; i++){
#ifdef TIME
	start_time = gettime();
#endif		
	/* copy data from host to device */
	//_clMemcpyH2D(devIn, fIn, szBufferByte);
	
	/* run on device of gpu */	
	//OCLRun8x8(devIn, devOut, uiImageW, uiImageH, 4);
	
	/* copy data d2h */	
	//_clMemcpyD2H(fOutOcl, devOut, szBufferByte);
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;
#endif	
	}
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif
	
	/* varify the output */
#ifdef 	VARIFY
	verify_array<float>(fOutOmp, fOutOcl, uiNumElems);
#endif	
		
	/* release context and gpu device memory */
	_clFree(devIn);
	_clFree(devOut);
	_clRelease();
	printf("ocl version on xpu\n\n");
	
	/* ---------------------
		ocl version on xpu (lm)
		---------------------*/	

#ifdef TIME	
	fprintf(fp, "\n");	
	fclose(fp);
#endif		
	
}
	catch(string msg)
	{
		printf("ERR:%s\n", msg.c_str());
		printf("Error catched\n");
	}

	if(uiIn!=NULL) free(uiIn);
	if(uiInB!=NULL) free(uiInB);
	if(uiOutOcl!=NULL) free(uiOutOcl);
	if(uiOutOmp!=NULL) free(uiOutOmp);
	return 1;
}


void OMPRunN(float *dst, float *src, uint N, int dir){

}


/*
	Reference impl. of DCT (native) on GPUs
*/
void OCLRunN(cl_mem devIn, cl_mem devOut, uint uiImageW, uint uiImageH, const int dir)
{

	uint uiBlkX = 32;
	uint uiBlkY = 1;
	uint uiGrpX = uiBlkX;
	uint uiGrpY = uiBlkY;
	uint uiRngX = uiImageW * uiImageH; //iDivUp(uiImageW, uiBlkX) * uiGrpX;
	uint uiRngY = 1; //iDivUp(uiImageH, uiBlkY) * uiGrpY;

	
	uint uiKD = dir;
	/* select a kernel id */
/*	switch(dir)
	{
		case DCT_FORWARD:
			uiKD = 3;
			break;
		case DCT_INVERSE:
			uiKD = 0;
			break;
		default:
			throw(string("unknown choice!\n"));
	}*/
	
	uint uiArgID = 0;
	_clSetArgs(uiKD, uiArgID++, devOut);
	_clSetArgs(uiKD, uiArgID++, devIn);
	_clSetArgs(uiKD, uiArgID++, &uiImageW, sizeof(uint));
	_clSetArgs(uiKD, uiArgID++, &uiImageH, sizeof(uint));
	_clSetArgs(uiKD, uiArgID++, &uiImageW, sizeof(uint));

	_clInvokeKernel2D(uiKD, uiRngX, uiRngY, uiGrpX, uiGrpY);
	
	return ;
}
