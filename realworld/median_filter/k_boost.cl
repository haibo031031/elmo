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

// RGB Median filter kernel using binary search method
// Uses 32 bit GMEM reads into a block of LMEM padded for apron of radius = 1 (3x3 neighbor op)
// R, G and B medians are treated separately 
//*****************************************************************************
__kernel void ckMedianBoost(__global uchar4* uc4Source, __global unsigned int* uiDest,
                       __local uchar4* uc4LocalData, int iLocalPixPitch, 
                       int iImageWidth, int iDevImageHeight
                       )
{
    // Get parent image x and y pixel coordinates from global ID, and compute offset into parent GMEM data
    int iImagePosX = get_global_id(0);
    int iDevYPrime = get_global_id(1) - 1;  // Shift offset up 1 radius (1 row) for reads
    int iDevGMEMOffset = mul24(iDevYPrime, (int)get_global_size(0)) + iImagePosX; 

    // Compute initial offset of current pixel within work group LMEM block
    int iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0) + 1;

    // Main read of GMEM data into LMEM
    if((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) && (iImagePosX < iImageWidth))
    {
        uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset];
    }
    else 
    {
        uc4LocalData[iLocalPixOffset] = (uchar4)0; 
    }

    // Work items with y ID < 2 read bottom 2 rows of LMEM 
    if (get_local_id(1) < 2)
    {
        // Increase local offset by 1 workgroup LMEM block height
        // to read in top rows from the next block region down
        iLocalPixOffset += mul24((int)get_local_size(1), iLocalPixPitch);

        // If source offset is within the image boundaries
        if (((iDevYPrime + get_local_size(1)) < iDevImageHeight) && (iImagePosX < iImageWidth))
        {
            // Read in top rows from the next block region down
            uc4LocalData[iLocalPixOffset] = uc4Source[iDevGMEMOffset + mul24(get_local_size(1), get_global_size(0))];
        }
        else 
        {
            uc4LocalData[iLocalPixOffset] = (uchar4)0; 
        }
    }

    // Work items with x ID at right workgroup edge will read Left apron pixel
    if (get_local_id(0) == (get_local_size(0) - 1))
    {
        // set local offset to read data from the next region over
        iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch);

        // If source offset is within the image boundaries and not at the leftmost workgroup
        if ((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) && (get_group_id(0) > 0))
        {
            // Read data into the LMEM apron from the GMEM at the left edge of the next block region over
            uc4LocalData[iLocalPixOffset] = uc4Source[mul24(iDevYPrime, (int)get_global_size(0)) + mul24(get_group_id(0), get_local_size(0)) - 1];
        }
        else 
        {
            uc4LocalData[iLocalPixOffset] = (uchar4)0; 
        }

        // If in the bottom 2 rows of workgroup block 
        if (get_local_id(1) < 2)
        {
            // Increase local offset by 1 workgroup LMEM block height
            // to read in top rows from the next block region down
            iLocalPixOffset += mul24((int)get_local_size(1), iLocalPixPitch);

            // If source offset in the next block down isn't off the image and not at the leftmost workgroup
            if (((iDevYPrime + get_local_size(1)) < iDevImageHeight) && (get_group_id(0) > 0))
            {
                // read in from GMEM (reaching down 1 workgroup LMEM block height and left 1 pixel)
                uc4LocalData[iLocalPixOffset] = uc4Source[mul24((iDevYPrime + (int)get_local_size(1)), (int)get_global_size(0)) + mul24(get_group_id(0), get_local_size(0)) - 1];
            }
            else 
            {
                uc4LocalData[iLocalPixOffset] = (uchar4)0; 
            }
        }
    } 
    else if (get_local_id(0) == 0) // Work items with x ID at left workgroup edge will read right apron pixel
    {
        // set local offset 
        iLocalPixOffset = mul24(((int)get_local_id(1) + 1), iLocalPixPitch) - 1;

        if ((iDevYPrime > -1) && (iDevYPrime < iDevImageHeight) && (mul24(((int)get_group_id(0) + 1), (int)get_local_size(0)) < iImageWidth))
        {
            // read in from GMEM (reaching left 1 pixel) if source offset is within image boundaries
            uc4LocalData[iLocalPixOffset] = uc4Source[mul24(iDevYPrime, (int)get_global_size(0)) + mul24((get_group_id(0) + 1), get_local_size(0))];
        }
        else 
        {
            uc4LocalData[iLocalPixOffset] = (uchar4)0; 
        }

        // Read bottom 2 rows of workgroup LMEM block
        if (get_local_id(1) < 2)
        {
            // increase local offset by 1 workgroup LMEM block height
            iLocalPixOffset += (mul24((int)get_local_size(1), iLocalPixPitch));

            if (((iDevYPrime + get_local_size(1)) < iDevImageHeight) && (mul24((get_group_id(0) + 1), get_local_size(0)) < iImageWidth) )
            {
                // read in from GMEM (reaching down 1 workgroup LMEM block height and left 1 pixel) if source offset is within image boundaries
                uc4LocalData[iLocalPixOffset] = uc4Source[mul24((iDevYPrime + (int)get_local_size(1)), (int)get_global_size(0)) + mul24((get_group_id(0) + 1), get_local_size(0))];
            }
            else 
            {
                uc4LocalData[iLocalPixOffset] = (uchar4)0; 
            }
        }
    }

    // Synchronize the read into LMEM
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute 
	// reset accumulators  
    float fMedianEstimate[3] = {128.0f, 128.0f, 128.0f};
    float fMinBound[3] = {0.0f, 0.0f, 0.0f};
    float fMaxBound[3] = {255.0f, 255.0f, 255.0f};

	// now find the median using a binary search - Divide and Conquer 256 gv levels for 8 bit plane
	for(int iSearch = 0; iSearch < 8; iSearch++)  // for 8 bit data, use 0..8.  For 16 bit data, 0..16. More iterations for more bits.
	{
        uint uiHighCount [3] = {0, 0, 0};
		
        // set local offset and kernel offset
        iLocalPixOffset = mul24((int)get_local_id(1), iLocalPixPitch) + get_local_id(0);

		// Row1 Left Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);					

		// Row1 Middle Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);					

		// Row1 Right Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset].z);					

		// set the offset into SMEM for next row
		iLocalPixOffset += (iLocalPixPitch - 2);	

		// Row2 Left Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);					

		// Row2 Middle Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);					

		// Row2 Right Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset].z);					

		// set the offset into SMEM for next row
		iLocalPixOffset += (iLocalPixPitch - 2);	

		// Row3 Left Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);					

		// Row3 Middle Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset++].z);					

		// Row3 Right Pix (RGB)
		uiHighCount[0] += (fMedianEstimate[0] < uc4LocalData[iLocalPixOffset].x);					
		uiHighCount[1] += (fMedianEstimate[1] < uc4LocalData[iLocalPixOffset].y);					
		uiHighCount[2] += (fMedianEstimate[2] < uc4LocalData[iLocalPixOffset].z);					

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

    // Write out to GMEM with restored offset
    if((iDevYPrime < iDevImageHeight) && (iImagePosX < iImageWidth))
    {
        uiDest[iDevGMEMOffset + get_global_size(0)] = uiPackedPix;
    }
}
