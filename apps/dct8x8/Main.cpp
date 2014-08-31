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
void GPURun(cl_mem devIn, cl_mem devOut, uint uiImageW, uint uiImageH, const int dir);
void CPURun(float *dst, float *src, uint stride, uint imageH, uint imageW, int dir);
void OCLRunN(cl_mem devIn, cl_mem devOut, uint uiImageW, uint uiImageH, const int dir);
void OMPRunN(float *dst, float *src, uint numElems, int dir);
void OMPRun8x8(float * dst, float * src, uint W, uint H, int dir);
void OCLRun8x8(cl_mem devIn, cl_mem devOut, uint uiImageW, uint uiImageH, const int dir);

static inline uint iDivUp(uint dividend, uint divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

#define DCT_FORWARD 666
#define DCT_INVERSE 777
#define BLOCK_SIZE 8
#define NUM_THREADS 24

int main(int argc, char ** argv)
{
	float * fIn = NULL, * fOutOcl = NULL, * fOutOmp = NULL;
	cl_mem devIn = NULL, devOut = NULL;
	
	
try{
	_clParseCommandLine(argc, argv);
	/* parameters */
	const int dir = DCT_FORWARD;	
	uint uiImageW = 8, uiImageH = 8, uiStride = 8;
	uint sz = atoi(argv[1]);
	string strSubfix = string(argv[2]);
	uiImageW = uiImageH = uiStride = sz;
	/* allocate memory space on the host */
	uint uiNumElems = uiImageW * uiImageH;
	uint szBufferByte = uiNumElems * sizeof(float);
	fIn = (float *)malloc(szBufferByte);
	fOutOcl = (float *)malloc(szBufferByte);
	fOutOmp = (float *)malloc(szBufferByte);
	
	/* initialization */
	fill<float>(fIn, uiNumElems, 255);		
	//float \
	fInC[64] = {-76, -73, -67, -62, -58, -67, -64, -55, \
				-65, -69, -73, -38, -19, -43, -59, -56, \
				-66, -69, -60, -15,  16, -24, -62, -55, \
				-65, -70, -57,  -6,  26, -22, -58, -59, \
				-61, -67, -60, -24,  -2, -40, -60, -58, \
				-49, -63, -68, -58, -51, -60, -70, -53, \
				-43, -57, -64, -69, -73, -67, -63, -45, \
				-41, -49, -59, -60, -63, -52, -50, -34};
	//float fOutC[64];
	const uint iter = 10;
	//OMPRun8x8(fOutOmp, fIn, uiImageW, uiImageH, 0);

	/*for(int h=0; h<uiImageH; h++)
	{
		for(int w=0; w<uiImageW; w++)
		{
			printf("%f\t", fOutC[h * uiImageW + w]);
		}
		printf("\n\n");		
	}
	throw(string("manually interuption\n"));
	*/ 
	
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
	
	/**************************CPUversusGPU****************************/
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
	OMPRun8x8(fOutOmp, fIn, uiImageW, uiImageH, dir);
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
	devIn = _clMalloc(szBufferByte);
	devOut = _clMalloc(szBufferByte);
#ifdef TIME
	deltaT = 0.0;
#endif		
	
	for(int i=0; i<iter; i++){
#ifdef TIME
	start_time = gettime();
#endif		
	/* copy data from host to device */
	_clMemcpyH2D(devIn, fIn, szBufferByte);
	
	/* run on device of gpu */	
	OCLRun8x8(devIn, devOut, uiImageW, uiImageH, 4);
	
	/* copy data d2h */	
	_clMemcpyD2H(fOutOcl, devOut, szBufferByte);
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
	printf("--------------------------\n");		
	/* device initialization and context setup */
	_clInit(platform_id, device_type, device_id);	

	/* create buffer on the device */
	devIn = _clMalloc(szBufferByte);
	devOut = _clMalloc(szBufferByte);
#ifdef TIME
	deltaT = 0.0;
#endif		
	
	for(int i=0; i<iter; i++){
#ifdef TIME
	start_time = gettime();
#endif		
	/* copy data from host to device */
	_clMemcpyH2D(devIn, fIn, szBufferByte);
	
	/* run on device of gpu */	
	OCLRun8x8(devIn, devOut, uiImageW, uiImageH, 5);
	
	/* copy data d2h */	
	_clMemcpyD2H(fOutOcl, devOut, szBufferByte);
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
	printf("ocl version on xpu (lm)\n\n");		

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

	if(fIn!=NULL) free(fIn);
	if(fOutOcl!=NULL) free(fOutOcl);
	if(fOutOmp!=NULL) free(fOutOmp);
	return 1;
}

/*
	DCT8x8 on CPU as reference
*/
////////////////////////////////////////////////////////////////////////////////
// Straightforward general-sized (i)DCT with O(N ** 2) complexity
// so that we don't forget what we're calculating :)
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979323846264338327950288f

void OMPRunN(float *dst, float *src, uint N, int dir){

    float *buf = (float *)malloc(N * sizeof(float));
#pragma omp parallel for num_threads(NUM_THREADS)
    for(uint k = 0; k < N; k++){
        buf[k] = 0;
        for(uint n = 0; n < N; n++)
            buf[k] += src[n] * cosf((float)PI / (float)N * ((float)n + 0.5f) * (float)k);
    }

    dst[0] = buf[0] * sqrtf(1.0f / (float)N);
    
#pragma omp parallel for num_threads(NUM_THREADS)
    for(uint i = 1; i < N; i++)
        dst[i] = buf[i] * sqrtf(2.0f / (float)N);

    free(buf);
}

/*
	dct8x8 - omp version
*/

void OMPRun8x8(float * dst, float * src, uint W, uint H, int dir)
{	
	float sqrt1d8 = sqrtf(1.0/8.0);
	float sqrt2d8 = sqrtf(2.0/8.0);
	float pid8 = (float)PI/8.0;
	
	for(uint v=0; v<H; v++)
	{
		for(uint u=0; u<W; u++)
		{
			/* for each output pixel */
			float au, av;			
			(u==0)?(au=sqrt1d8):(au=sqrt2d8);
			(v==0)?(av=sqrt1d8):(av=sqrt2d8);
			
			float res = 0.0f;
			uint Bx = u / BLOCK_SIZE; /* block number x */
			uint By = v / BLOCK_SIZE; /* block number y */
			uint Sx = BLOCK_SIZE; /* block size x */
			uint Sy = BLOCK_SIZE; /* block size y */			
			
			for(uint y=0; y<BLOCK_SIZE; y++)
			{
				for(uint x=0; x<BLOCK_SIZE; x++)
				{
					
					uint uiIIdx = (By * Sy + y) * W + Bx * Sx + x;
					res += au * av * src[uiIIdx] * cosf(pid8 * ((float)x + 0.5) * u) * cosf(pid8 * ((float)y + 0.5) * v);
				}
			}
			uint uiOIdx = v * W + u;
			dst[uiOIdx] = res;
		}	
	}
}

static void naiveIDCT(float *dst, float *src, uint N){
    float *buf = (float *)malloc(N * sizeof(float));

    for(uint k = 0; k < N; k++){
        buf[k] = sqrtf(0.5f) * src[0];
        for(uint n = 1; n < N; n++)
            buf[k] += src[n] * cosf(PI / (float)N * (float)n * ((float)k + 0.5f) );
    }

    for(uint i = 0; i < N; i++)
        dst[i] = buf[i] * sqrtf(2.0f / (float)N);

    free(buf);
}

////////////////////////////////////////////////////////////////////////////////
// Hardcoded unrolled fast 8-point (i)DCT
////////////////////////////////////////////////////////////////////////////////
#define C_a 1.3870398453221474618216191915664f       //a = sqrt(2) * cos(1 * pi / 16)
#define C_b 1.3065629648763765278566431734272f       //b = sqrt(2) * cos(2 * pi / 16)
#define C_c 1.1758756024193587169744671046113f       //c = sqrt(2) * cos(3 * pi / 16)
#define C_d 0.78569495838710218127789736765722f      //d = sqrt(2) * cos(5 * pi / 16)
#define C_e 0.54119610014619698439972320536639f      //e = sqrt(2) * cos(6 * pi / 16)
#define C_f 0.27589937928294301233595756366937f      //f = sqrt(2) * cos(7 * pi / 16)
#define C_norm 0.35355339059327376220042218105242f   //1 / sqrt(8)

static void DCT8(float *dst, float *src, uint ostride, uint istride){
    float X07P = src[0 * istride] + src[7 * istride];
    float X16P = src[1 * istride] + src[6 * istride];
    float X25P = src[2 * istride] + src[5 * istride];
    float X34P = src[3 * istride] + src[4 * istride];

    float X07M = src[0 * istride] - src[7 * istride];
    float X61M = src[6 * istride] - src[1 * istride];
    float X25M = src[2 * istride] - src[5 * istride];
    float X43M = src[4 * istride] - src[3 * istride];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    dst[0 * ostride] = C_norm * (X07P34PP + X16P25PP);
    dst[2 * ostride] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    dst[4 * ostride] = C_norm * (X07P34PP - X16P25PP);
    dst[6 * ostride] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    dst[1 * ostride] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    dst[3 * ostride] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    dst[5 * ostride] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    dst[7 * ostride] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

static void IDCT8(float *dst, float *src, uint ostride, uint istride){
    float Y04P   = src[0 * istride] + src[4 * istride];
    float Y2b6eP = C_b * src[2 * istride] + C_e * src[6 * istride];

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * src[7 * istride] + C_a * src[1 * istride] + C_c * src[3 * istride] + C_d * src[5 * istride];
    float Y7a1fM3d5cMP = C_a * src[7 * istride] - C_f * src[1 * istride] + C_d * src[3 * istride] - C_c * src[5 * istride];

    float Y04M   = src[0*istride] - src[4*istride];
    float Y2e6bM = C_e * src[2*istride] - C_b * src[6*istride];

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * src[1 * istride] - C_d * src[7 * istride] - C_f * src[3 * istride] - C_a * src[5 * istride];
    float Y1d7cP3a5fMM = C_d * src[1 * istride] + C_c * src[7 * istride] - C_a * src[3 * istride] + C_f * src[5 * istride];

    dst[0 * ostride] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    dst[7 * ostride] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    dst[4 * ostride] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    dst[3 * ostride] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    dst[1 * ostride] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    dst[5 * ostride] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    dst[2 * ostride] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    dst[6 * ostride] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

void CPURun(float *dst, float *src, uint stride, uint imageH, uint imageW, int dir){
    assert( (dir == DCT_FORWARD) || (dir == DCT_INVERSE) );

    for (uint i = 0; i + BLOCK_SIZE - 1 < imageH; i += BLOCK_SIZE){
        for (uint j = 0; j + BLOCK_SIZE - 1 < imageW; j += BLOCK_SIZE){
            //process rows
            for(uint k = 0; k < BLOCK_SIZE; k++)
                if(dir == DCT_FORWARD)
                    DCT8(dst + (i + k) * stride + j, src + (i + k) * stride + j, 1, 1);
                else
                    IDCT8(dst + (i + k) * stride + j, src + (i + k) * stride + j, 1, 1);

            //process columns
            for(uint k = 0; k < BLOCK_SIZE; k++)
                if(dir == DCT_FORWARD)
                    DCT8(dst + i * stride + (j + k), dst + i * stride + (j + k), stride, stride);
                else
                    IDCT8(dst + i * stride + (j + k), dst + i * stride + (j + k), stride, stride);
        }
    }
}

/*
	Reference impl. of BoxFilter on GPUs
*/
void GPURun(cl_mem devIn, cl_mem devOut, uint uiImageW, uint uiImageH, const int dir)
{

	uint uiBlkX = 32;
	uint uiBlkY = 16;
	uint uiGrpX = uiBlkX;
	uint uiGrpY = uiBlkY/BLOCK_SIZE;
	uint uiRngX = iDivUp(uiImageW, uiBlkX) * uiGrpX;
	uint uiRngY = iDivUp(uiImageH, uiBlkY) * uiGrpY;

	
	uint uiKD = 0;
	/* select a kernel id */
	switch(dir)
	{
		case DCT_FORWARD:
			uiKD = 0;
			break;
		case DCT_INVERSE:
			uiKD = 1;
			break;
		default:
			throw(string("unknown choice!\n"));
	}
	
	uint uiArgID = 0;
	_clSetArgs(uiKD, uiArgID++, devOut);
	_clSetArgs(uiKD, uiArgID++, devIn);
	_clSetArgs(uiKD, uiArgID++, &uiImageW, sizeof(uint));
	_clSetArgs(uiKD, uiArgID++, &uiImageH, sizeof(uint));
	_clSetArgs(uiKD, uiArgID++, &uiImageW, sizeof(uint));

	_clInvokeKernel2D(uiKD, uiRngX, uiRngY, uiGrpX, uiGrpY);
	
	return ;
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

/*
	Reference impl. of DCT8x8 (native) on OCL platforms
*/
void OCLRun8x8(cl_mem devIn, cl_mem devOut, uint uiImageW, uint uiImageH, const int dir)
{

	//uint uiBlkX = 32;
	//uint uiBlkY = 1;
	uint uiGrpX = 8;
	uint uiGrpY = 8;
	uint uiRngX = iDivUp(uiImageW, uiGrpX) * uiGrpX;
	uint uiRngY = iDivUp(uiImageH, uiGrpY) * uiGrpY;

	
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
	//_clSetArgs(uiKD, uiArgID++, &dir, sizeof(uint));

	_clInvokeKernel2D(uiKD, uiRngX, uiRngY, uiGrpX, uiGrpY);
	
	return ;
}


