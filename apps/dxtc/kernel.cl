  /*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define BLOCK_SIZE 8

////////////////////////////////////////////////////////////////////////////////
// Hardcoded unrolled fast 8-point (i)DCT routines
////////////////////////////////////////////////////////////////////////////////
#define    C_a 1.3870398453221474618216191915664f  //a = sqrt(2) * cos(1 * pi / 16)
#define    C_b 1.3065629648763765278566431734272f  //b = sqrt(2) * cos(2 * pi / 16)
#define    C_c 1.1758756024193587169744671046113f  //c = sqrt(2) * cos(3 * pi / 16)
#define    C_d 0.78569495838710218127789736765722f //d = sqrt(2) * cos(5 * pi / 16)
#define    C_e 0.54119610014619698439972320536639f //e = sqrt(2) * cos(6 * pi / 16)
#define    C_f 0.27589937928294301233595756366937f //f = sqrt(2) * cos(7 * pi / 16)
#define C_norm 0.35355339059327376220042218105242f //1 / sqrt(8)

inline void DCT8(float *D){
    float X07P = D[0] + D[7];
    float X16P = D[1] + D[6];
    float X25P = D[2] + D[5];
    float X34P = D[3] + D[4];

    float X07M = D[0] - D[7];
    float X61M = D[6] - D[1];
    float X25M = D[2] - D[5];
    float X43M = D[4] - D[3];

    float X07P34PP = X07P + X34P;
    float X07P34PM = X07P - X34P;
    float X16P25PP = X16P + X25P;
    float X16P25PM = X16P - X25P;

    D[0] = C_norm * (X07P34PP + X16P25PP);
    D[2] = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
    D[4] = C_norm * (X07P34PP - X16P25PP);
    D[6] = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

    D[1] = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
    D[3] = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
    D[5] = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
    D[7] = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

inline void IDCT8(float *D){
    float Y04P   = D[0] + D[4];
    float Y2b6eP = C_b * D[2] + C_e * D[6];

    float Y04P2b6ePP = Y04P + Y2b6eP;
    float Y04P2b6ePM = Y04P - Y2b6eP;
    float Y7f1aP3c5dPP = C_f * D[7] + C_a * D[1] + C_c * D[3] + C_d * D[5];
    float Y7a1fM3d5cMP = C_a * D[7] - C_f * D[1] + C_d * D[3] - C_c * D[5];

    float Y04M   = D[0] - D[4];
    float Y2e6bM = C_e * D[2] - C_b * D[6];

    float Y04M2e6bMP = Y04M + Y2e6bM;
    float Y04M2e6bMM = Y04M - Y2e6bM;
    float Y1c7dM3f5aPM = C_c * D[1] - C_d * D[7] - C_f * D[3] - C_a * D[5];
    float Y1d7cP3a5fMM = C_d * D[1] + C_c * D[7] - C_a * D[3] + C_f * D[5];

    D[0] = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
    D[7] = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
    D[4] = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
    D[3] = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

    D[1] = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
    D[5] = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
    D[2] = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
    D[6] = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}



////////////////////////////////////////////////////////////////////////////////
// 8x8 DCT kernels
////////////////////////////////////////////////////////////////////////////////
#define BLOCK_X 32
#define BLOCK_Y 16

__kernel __attribute__((reqd_work_group_size(BLOCK_X, BLOCK_Y / BLOCK_SIZE, 1)))
void DCT8x8(
    __global float *d_Dst,
    __global float *d_Src,
    uint stride,
    uint imageH,
    uint imageW
){
    __local float l_Transpose[BLOCK_Y][BLOCK_X + 1];
    const uint    localX = get_local_id(0);
    const uint    localY = BLOCK_SIZE * get_local_id(1);
    const uint modLocalX = localX & (BLOCK_SIZE - 1);
    const uint   globalX = get_group_id(0) * BLOCK_X + localX;
    const uint   globalY = get_group_id(1) * BLOCK_Y + localY;

    //Process only full blocks
    if( (globalX - modLocalX + BLOCK_SIZE - 1 >= imageW) || (globalY + BLOCK_SIZE - 1 >= imageH) )
        return;

    __local float *l_V = &l_Transpose[localY +         0][localX +         0];
    __local float *l_H = &l_Transpose[localY + modLocalX][localX - modLocalX];
    d_Src += globalY * stride + globalX;
    d_Dst += globalY * stride + globalX;

    float D[8];
    for(uint i = 0; i < BLOCK_SIZE; i++)
        l_V[i * (BLOCK_X + 1)] = d_Src[i * stride];

    for(uint i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_H[i];
    DCT8(D);
    for(uint i = 0; i < BLOCK_SIZE; i++)
        l_H[i] = D[i];

    for(uint i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_V[i * (BLOCK_X + 1)];
    DCT8(D);

    for(uint i = 0; i < BLOCK_SIZE; i++)
        d_Dst[i * stride] = D[i];
}

__kernel __attribute__((reqd_work_group_size(BLOCK_X, BLOCK_Y / BLOCK_SIZE, 1)))
void IDCT8x8(
    __global float *d_Dst,
    __global float *d_Src,
    uint stride,
    uint imageH,
    uint imageW
){
    __local float l_Transpose[BLOCK_Y][BLOCK_X + 1];
    const uint    localX = get_local_id(0);
    const uint    localY = BLOCK_SIZE * get_local_id(1);
    const uint modLocalX = localX & (BLOCK_SIZE - 1);
    const uint   globalX = get_group_id(0) * BLOCK_X + localX;
    const uint   globalY = get_group_id(1) * BLOCK_Y + localY;

    //Process only full blocks
    if( (globalX - modLocalX + BLOCK_SIZE - 1 >= imageW) || (globalY + BLOCK_SIZE - 1 >= imageH) )
        return;

    __local float *l_V = &l_Transpose[localY +         0][localX +         0];
    __local float *l_H = &l_Transpose[localY + modLocalX][localX - modLocalX];
    d_Src += globalY * stride + globalX;
    d_Dst += globalY * stride + globalX;

    float D[8];
    for(uint i = 0; i < BLOCK_SIZE; i++)
        l_V[i * (BLOCK_X + 1)] = d_Src[i * stride];

    for(uint i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_H[i];
    IDCT8(D);
    for(uint i = 0; i < BLOCK_SIZE; i++)
        l_H[i] = D[i];

    for(uint i = 0; i < BLOCK_SIZE; i++)
        D[i] = l_V[i * (BLOCK_X + 1)];
    IDCT8(D);
    for(uint i = 0; i < BLOCK_SIZE; i++)
        d_Dst[i * stride] = D[i];
}

/*
	native DCT8x8
*/
#define PI 3.14159265358979323846264338327950288f
__kernel void native(__global float * dst,
    __global float * src,
    uint stride,
    uint imageH,
    uint imageW)
{
	const uint gId = get_global_id(0);
	//const uint gIdY = get_global_id(1);	
	//const uint gId = gIdY * stride + gIdX;
	const uint N = imageH * imageW;
	dst[gId] = 0.0f;
	for(int n=0; n<N; n++)
	{
		dst[gId] += src[n] * cos((float)PI/(float)N * ((float)n+0.5f) * (float)gId);
	}
	
	if(gId==0)
		dst[gId] = dst[gId] * sqrt(1.0f/(float)N);
	else
		dst[gId] = dst[gId] * sqrt(2.0f/(float)N);
}

#define S 32

__kernel void nativeLM(__global float * dst,
    __global float * src,
    uint stride,
    uint imageH,
    uint imageW)
{
	__local float LM[S];
	const uint gId = get_global_id(0);
	const uint lId = get_local_id(0);
	//const uint gIdY = get_global_id(1);	
	//const uint gId = gIdY * stride + gIdX;
	const uint N = imageH * imageW;
	dst[gId] = 0.0f;
	for(int n=0; n<N; n=n+S)
	{
		LM[lId] = src[n+lId];
		barrier(CLK_LOCAL_MEM_FENCE);
		
		for(int i=0; i<S; i++)
		{
			//dst[gId] += src[n] * cos((float)PI/(float)N * ((float)n+0.5f) * (float)gId);
			dst[gId] += LM[i] * cos((float)PI/(float)N * ((float)(n+i)+0.5f) * (float)gId);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if(gId==0)
		dst[gId] = dst[gId] * sqrt(1.0f/(float)N);
	else
		dst[gId] = dst[gId] * sqrt(2.0f/(float)N);
}

__kernel void OCLRun8x8(__global float * dst, \
				const __global float * src, \
				uint W, uint H)
{	
	float sqrt1d8 = sqrt(1.0f/8.0f);
	float sqrt2d8 = sqrt(2.0f/8.0f);
	float pid8 = (float)PI/8.0f;
	uint v = get_global_id(1);
	uint u = get_global_id(0);
	if((v<H) && (u<W)){
	
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
				res += au * av * src[uiIIdx] * cos(pid8 * ((float)x + 0.5f) * (float)u) * cos(pid8 * ((float)y + 0.5f) * (float)v);
			}
		}
		uint uiOIdx = v * W + u;
		dst[uiOIdx] = res;
	}
}

#define WG 8
__kernel void OCLRun8x8LM(__global float * dst, \
				const __global float * src, \
				uint W, uint H)
{	
	__local float LM[WG][WG];
	float sqrt1d8 = sqrt(1.0f/8.0f);
	float sqrt2d8 = sqrt(2.0f/8.0f);
	float pid8 = (float)PI/8.0f;
	uint v = get_global_id(1);
	uint u = get_global_id(0);
	uint m = get_local_id(0);
	uint n = get_local_id(1);
	
	if((v<H) && (u<W)){
	
		/* for each output pixel */
		float au, av;			
		(u==0)?(au=sqrt1d8):(au=sqrt2d8);
		(v==0)?(av=sqrt1d8):(av=sqrt2d8);
			
		float res = 0.0f;
		uint Bx = u / BLOCK_SIZE; /* block number x */
		uint By = v / BLOCK_SIZE; /* block number y */
		uint Sx = BLOCK_SIZE; /* block size x */
		uint Sy = BLOCK_SIZE; /* block size y */			
		
		/* load data elements into local memory */	
		LM[n][m] = src[v * W + u];
		barrier(CLK_LOCAL_MEM_FENCE);
		/* using the data elements in the local memory */
		for(uint y=0; y<BLOCK_SIZE; y++)
		{
			for(uint x=0; x<BLOCK_SIZE; x++)
			{				
				//uint uiIIdx = y * BLOCK_SIZE + x;
				//res += au * av * src[uiIIdx] * cos(pid8 * ((float)x + 0.5f) * (float)u) * cos(pid8 * ((float)y + 0.5f) * (float)v);
				res += au * av * LM[y][x] * cos(pid8 * ((float)x + 0.5f) * (float)u) * cos(pid8 * ((float)y + 0.5f) * (float)v);
			}
		}
		uint uiOIdx = v * W + u;
		dst[uiOIdx] = res;
	}
}
