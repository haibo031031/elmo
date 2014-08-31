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
#define VARIFY
#endif


#define BLOCK_SIZE 16

void CPURun(unsigned int* uiIn, unsigned int* uiOut, unsigned int uiW, unsigned int uiH);
void mf_1(cl_mem d_uiIn, cl_mem d_uiOut, uint uiW, uint uiH);
void mf_2(cl_mem d_uiIn, cl_mem d_uiOut, uint uiW, uint uiH);

int main(int argc, char ** argv)
{
	unsigned int * uiIn = NULL, * uiOut = NULL, * uiOutRef = NULL;
	cl_mem d_uiIn = NULL, d_uiOut = NULL;
try{
	if(argc!=2){
		printf("need 1 parameter here!!!");
		exit(-1);
	}

	_clInit(1, "gpu", 0);
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
	
	// parameters
	uint side = atoi(argv[1]);
	uint uiW = side;
	uint uiH = side;
	uint size = uiW * uiH;

	printf("wData=%d, hData=%d\n", uiW, uiH);
	
	// allocate memory space on the host and device side
	uiIn = (uint * )malloc(size * sizeof(uint));
	uiOut = (uint * )malloc(size * sizeof(uint));
	uiOutRef = (uint * )malloc(size * sizeof(uint));
	
	d_uiIn = _clMalloc(size * sizeof(cl_uint));	
	d_uiOut = _clMalloc(size * sizeof(cl_uint));	

	// initialization
	fill<uint>(uiIn, size, 16);
	fill<uint>(uiOut, size, 16);

	// copy data from host to device
	_clMemcpyH2D(d_uiIn, uiIn, size * sizeof(uint));
	
	// warm-up
	mf_1(d_uiIn, d_uiOut, uiW, uiH);
	
#ifdef VARIFY	
	CPURun(uiIn, uiOutRef, uiW, uiH);
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

		mf_1(d_uiIn, d_uiOut, uiW, uiH);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(uiOut, d_uiOut, size * sizeof(uint));
	verify_array_int<uint>(uiOut, uiOutRef, size);
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

		mf_2(d_uiIn, d_uiOut, uiW, uiH);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(uiOut, d_uiOut, size * sizeof(uint));
	verify_array_int<uint>(uiOut, uiOutRef, size);
#endif //VARIFY

#ifdef TIME	
	fprintf(fp, "\n");	
	fclose(fp);
#endif	
}
	catch(string msg)
	{
		printf("ERR:%s\n", msg.c_str());
	}

	_clFree(d_uiIn);
	_clFree(d_uiOut);
	_clRelease();
	if(uiIn!=NULL) free(uiIn);
	if(uiOut!=NULL) free(uiOut);
	if(uiOutRef!=NULL) free(uiOutRef);

	return 1;
}

//*****************************************************************
//! Exported Host/C++ RGB 3x3 Median function
//! Gradient intensity is from RSS combination of H and V gradient components
//! R, G and B medians are treated separately 
//!
//! @param uiInputImage     pointer to input data
//! @param uiOutputImage    pointer to output dataa
//! @param uiWidth          width of image
//! @param uiHeight         height of image
//*****************************************************************
void CPURun(unsigned int* uiInputImage, unsigned int* uiOutputImage, 
                                   unsigned int uiWidth, unsigned int uiHeight)
{
	// do the Median 
	for(unsigned int y = 0; y < uiHeight; y++)			// all the rows
	{
		for(unsigned int x = 0; x < uiWidth; x++)		// all the columns
		{
            // local registers for working with RGB subpixels and managing border
            unsigned char* ucRGBA; 
            const unsigned int uiZero = 0U;

		    // reset accumulators  
            float fMedianEstimate [3] = {128.0f, 128.0f, 128.0f};
            float fMinBound [3]= {0.0f, 0.0f, 0.0f};
            float fMaxBound[3] = {255.0f, 255.0f, 255.0f};

		    // now find the median using a binary search - Divide and Conquer 256 gv levels for 8 bit plane
		    for(int iSearch = 0; iSearch < 8; iSearch++)  
		    {
                unsigned int uiHighCount[3] = {0,0,0};

			    for (int iRow = -1; iRow <= 1 ; iRow++)
			    {
                    int iLocalOffset = (int)((iRow + y) * uiWidth) + x - 1;

				    // Left Pix (RGB)
                    // Read in pixel value to local register:  if boundary pixel, use zero
                    if ((x > 0) && ((y + iRow) >= 0) && ((y + iRow) < uiHeight))
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	

				    // Middle Pix (RGB)
                    // Increment offset and read in next pixel value to a local register:  if boundary pixel, use zero
                    iLocalOffset++;
                    if (((y + iRow) >= 0) && ((y + iRow) < uiHeight)) 
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	

				    // Right Pix (RGB)
                    // Increment offset and read in next pixel value to a local register:  if boundary pixel, use zero
                    iLocalOffset++;
                    if ((x < (uiWidth - 1)) && ((y + iRow) >= 0) && ((y + iRow) < uiHeight))
                    {
                        ucRGBA = (unsigned char*)&uiInputImage [iLocalOffset];
                    }
                    else 
                    {
                        ucRGBA = (unsigned char*)&uiZero;
                    }
				    uiHighCount[0] += (fMedianEstimate[0] < ucRGBA[0]);					
				    uiHighCount[1] += (fMedianEstimate[1] < ucRGBA[1]);					
				    uiHighCount[2] += (fMedianEstimate[2] < ucRGBA[2]);	
			    }

			    //********************************
			    // reset the appropriate bound, depending upon counter
			    if(uiHighCount[0] > 4)
			    {
				    fMinBound[0] = fMedianEstimate[0];				
			    }
			    else
			    {
				    fMaxBound[0] = fMedianEstimate[0];				
			    }

			    if(uiHighCount[1] > 4)
			    {
				    fMinBound[1] = fMedianEstimate[1];				
			    }
			    else
			    {
				    fMaxBound[1] = fMedianEstimate[1];				
			    }

			    if(uiHighCount[2] > 4)
			    {
				    fMinBound[2] = fMedianEstimate[2];				
			    }
			    else
			    {
				    fMaxBound[2] = fMedianEstimate[2];				
			    }

			    // refine the estimate
			    fMedianEstimate[0] = 0.5f * (fMaxBound[0] + fMinBound[0]);
			    fMedianEstimate[1] = 0.5f * (fMaxBound[1] + fMinBound[1]);
			    fMedianEstimate[2] = 0.5f * (fMaxBound[2] + fMinBound[2]);
		    }

            // pack into a monochrome uint 
            unsigned int uiPackedPix = 0x000000FF & (unsigned int)(fMedianEstimate[0] + 0.5f);
            uiPackedPix |= 0x0000FF00 & (((unsigned int)(fMedianEstimate[1] + 0.5f)) << 8);
            uiPackedPix |= 0x00FF0000 & (((unsigned int)(fMedianEstimate[2] + 0.5f)) << 16);

			// copy to output
			uiOutputImage[y * uiWidth + x] = uiPackedPix;	
			printf("%d\t", uiPackedPix);
		}
	}

    return ;
}

/*
	Impl. of MediumFilter on GPUs from NV SDK
*/

void mf_1(cl_mem d_uiIn, cl_mem d_uiOut, uint uiW, uint uiH)
{
	uint range_x = uiW;
	uint range_y = uiH;
	uint group_x = BLOCK_SIZE;
	uint group_y = BLOCK_SIZE;
	int iLocalPixPitch = group_x + 2;
	
	uint kernel_id = 0;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_uiIn);
	_clSetArgs(kernel_id, arg_idx++, d_uiOut);
	_clSetArgs(kernel_id, arg_idx++, NULL, iLocalPixPitch * (group_y + 2) * sizeof(cl_uint));
	_clSetArgs(kernel_id, arg_idx++, &iLocalPixPitch, sizeof(cl_int));
	_clSetArgs(kernel_id, arg_idx++, &uiW, sizeof(cl_uint));
	_clSetArgs(kernel_id, arg_idx++, &uiH, sizeof(cl_uint));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

/*

	Impl. of MediumFilter on GPUs using CLMemBoost
*/

void mf_2(cl_mem d_uiIn, cl_mem d_uiOut, uint uiW, uint uiH)
{
	uint range_x = uiW;
	uint range_y = uiH;
	uint group_x = BLOCK_SIZE;
	uint group_y = BLOCK_SIZE;
	int iLocalPixPitch = group_x + 2;
	
	uint kernel_id = 1;
	uint arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_uiIn);
	_clSetArgs(kernel_id, arg_idx++, d_uiOut);
	_clSetArgs(kernel_id, arg_idx++, NULL, iLocalPixPitch * (group_y + 2) * sizeof(cl_uint));
	_clSetArgs(kernel_id, arg_idx++, &iLocalPixPitch, sizeof(cl_int));
	_clSetArgs(kernel_id, arg_idx++, &uiW, sizeof(cl_uint));
	_clSetArgs(kernel_id, arg_idx++, &uiH, sizeof(cl_uint));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
