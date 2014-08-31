// Inline device function to convert 32-bit unsigned integer to floating point rgba color 
//*****************************************************************
float4 rgbaUintToFloat4(unsigned int c)
{
    float4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

// Inline device function to convert floating point rgba color to 32-bit unsigned integer
//*****************************************************************
unsigned int rgbaFloat4ToUint(float4 rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
    return uiPackedPix;
}

// Row summation filter kernel with rescaling, using Image (texture)
// USETEXTURE switch passed in via OpenCL clBuildProgram call options string at app runtime
//*****************************************************************
#ifdef USETEXTURE
    // Row summation filter kernel with rescaling, using Image (texture)
    __kernel void BoxRowsTex( __read_only image2d_t SourceRgbaTex, __global unsigned int* uiDest, sampler_t RowSampler, 
                              unsigned int uiWidth, unsigned int uiHeight, int iRadius, float fScale)
    {

    }
#endif

// Row summation filter kernel with rescaling, using LMEM
// USELMEM switch passed in via OpenCL clBuildProgram call options string at app runtime
//*****************************************************************
    // Row summation filter kernel with rescaling, using LMEM
    __kernel void BoxRowsLmem( __global const uchar4* uc4Source, __global unsigned int* uiDest,
                               __local uchar4* uc4LocalData,
                               unsigned int uiWidth, unsigned int uiHeight, int iRadius, int iRadiusAligned, 
                               float fScale, unsigned int uiNumOutputPix)
    {
        // Compute x and y pixel coordinates from group ID and local ID indexes
        int globalPosX = ((int)get_group_id(0) * uiNumOutputPix) + (int)get_local_id(0) - iRadiusAligned;
        int globalPosY = (int)get_group_id(1);
        int iGlobalOffset = globalPosY * uiWidth + globalPosX;

        // Read global data into LMEM
        if (globalPosX >= 0 && globalPosX < uiWidth)
        {
            uc4LocalData[get_local_id(0)] = uc4Source[iGlobalOffset];
        }
        else 
        {
            uc4LocalData[get_local_id(0)].xyzw = (uchar4)0; 
        }

        // Synchronize the read into LMEM
        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute (if pixel plus apron is within bounds)
        if((globalPosX >= 0) && (globalPosX < uiWidth) && (get_local_id(0) >= iRadiusAligned) && (get_local_id(0) < (iRadiusAligned + (int)uiNumOutputPix)))
        {
            // Init summation registers to zero
            float4 f4Sum = (float4)0.0f;

            // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
            int iOffsetX = (int)get_local_id(0) - iRadius;
            int iLimit = iOffsetX + (2 * iRadius) + 1;
            for(iOffsetX; iOffsetX < iLimit; iOffsetX++)
            {
                f4Sum.x += uc4LocalData[iOffsetX].x;
                f4Sum.y += uc4LocalData[iOffsetX].y;
                f4Sum.z += uc4LocalData[iOffsetX].z;
                f4Sum.w += uc4LocalData[iOffsetX].w; 
            }

            // Use inline function to scale and convert registers to packed RGBA values in a uchar4, and write back out to GMEM
            uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
        }
    }

// Column kernel using coalesced global memory reads
//*****************************************************************
__kernel void BoxColumns(__global unsigned int* uiInputImage, __global unsigned int* uiOutputImage, 
                         unsigned int uiWidth, unsigned int uiHeight, int iRadius, float fScale)
{
	size_t globalPosX = get_global_id(0);
    uiInputImage = &uiInputImage[globalPosX];
    uiOutputImage = &uiOutputImage[globalPosX];

    // do left edge
    float4 f4Sum;
    f4Sum = rgbaUintToFloat4(uiInputImage[0]) * (float4)(iRadius);
    for (int y = 0; y < iRadius + 1; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
    }
    uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);
    for(int y = 1; y < iRadius + 1; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[0]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
    
    // main loop
    for(int y = iRadius + 1; y < uiHeight - iRadius; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }

    // do right edge
    for (int y = uiHeight - iRadius; y < uiHeight; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
}

/*
	Fill local memory in the FCTH way.

	_clMemWriteBRO_FCTH(uc4Source, uiWidth, uiHeight, 0, iRadius, uc4LocalData, 0);
*/
void _clMemWriteBRO_FCTH(__global uchar4 * in, uint w, uint h, uint base_addr, int r, __local uchar4 * cache, uint p)
{
	uchar4 val = 0;
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_gl_idx = t_gl_y * w + t_gl_x;
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	
	uint length_x = wg_x + 2 * r + p;
	uint length_y = wg_y + 2 * r;
	// 1. central part
	int s_lc_x = t_lc_x + r;
	int s_lc_y = t_lc_y + r;
	uint s_gl_x = t_lc_x, s_gl_y = t_lc_y;	
	cache[s_lc_y * length_x + s_lc_x] = in[base_addr + s_gl_y * w + s_gl_x];
	if(r>0)
	{
		// 2-1 top-left		
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x - r, s_gl_y = t_gl_y - r;
			//if(s_gl_x<0) s_gl_x = 0;
			//if(s_gl_y<0) s_gl_y = 0;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if((s_gl_x < 0) || (s_gl_y < 0))  val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}
		// 2-2 top
		if(t_lc_y<r)
		{
			s_lc_x = t_lc_x + r, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x, s_gl_y = t_gl_y - r;
			//if(s_gl_y<0) s_gl_y = 0;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];		
			if(s_gl_y<0)	val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}
		// 2-3 top-right
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x + r + wg_x, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x + wg_x, s_gl_y = t_gl_y - r;
			//if(s_gl_x>=w) s_gl_x = w-1;
			//if(s_gl_y<0) s_gl_y = 0;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if((s_gl_x>=w)||(s_gl_y<0))	val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}
		// 2-4 bottom-left
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x, s_lc_y = t_lc_y + r + wg_y;
			s_gl_x = t_gl_x - r, s_gl_y = t_gl_y + wg_y;
			//if(s_gl_x<0) s_gl_x = 0;
			//if(s_gl_y>=h) s_gl_y = h-1;

			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if((s_gl_x<0)||(s_gl_y>=h)) val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;

		}	
		// 2-5 bottom
		if(t_lc_y<r)
		{

			s_lc_x = t_lc_x + r, s_lc_y = t_lc_y + r + wg_y;
			s_gl_x = t_gl_x, s_gl_y = t_gl_y + wg_y;
			//if(s_gl_y>=h) s_gl_y = h-1;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];

			if(s_gl_y>=h) val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}	
		// 2-6 bottom-right

		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x + r + wg_x, s_lc_y = t_lc_y + r + wg_y;
			s_gl_x = t_gl_x + wg_x, s_gl_y = t_gl_y + wg_y;

			//if(s_gl_x>=w) s_gl_x = w-1;
			//if(s_gl_y>=h) s_gl_y = h-1;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if((s_gl_x>=w)||(s_gl_y>=h)) val = 0;

			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}	
		// 2-7 left
		if(t_lc_x<r)

		{
			s_lc_x = t_lc_x, s_lc_y = t_lc_y + r;
			s_gl_x = t_gl_x - r, s_gl_y = t_gl_y;
			//if(s_gl_x<0) s_gl_x = 0;

			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if(s_gl_x<0) val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;

		}	
		// 2-8 right
		if(t_lc_x<r)
		{
			s_lc_x = t_lc_x + r + wg_x, s_lc_y = t_lc_y + r;

			s_gl_x = t_gl_x + wg_x, s_gl_y = t_gl_y;
			//if(s_gl_x>=w) s_gl_x = w-1;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if(s_gl_x>=w) val = 0;

			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}

/*
	Fill local memory in the FCTH way.

	_clMemWriteBRO_FCTH(uc4Source, uiWidth, uiHeight, 0, iRadius, uc4LocalData, 0);
*/
void _clMemWriteBRO_FCTH1d(__global uchar4 * in, uint w, uint h, uint base_addr, int r, __local uchar4 * cache, uint p)
{
	uchar4 val = 0;
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_gl_idx = t_gl_y * w + t_gl_x;
	uint t_lc_x = get_local_id(0);
	//uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	//uint wg_y = get_local_size(1);
	
	uint length_x = wg_x + 2 * r + p;
	//uint length_y = wg_y + 2 * r;
	// 1. central part
	int s_lc_x = t_lc_x + r;
	//int s_lc_y = (t_lc_y==0)?(0):(t_lc_y + r);
	//uint s_gl_x = t_lc_x, s_gl_y = t_lc_y;
	uint s_gl_x = t_lc_x;		
	//cache[s_lc_y * length_x + s_lc_x] = in[base_addr + s_gl_y * w + s_gl_x];
	cache[s_lc_x] = in[t_gl_idx];
	if(r>0)
	{
		/*// 2-1 top-left		
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x - r, s_gl_y = t_gl_y - r;
			//if(s_gl_x<0) s_gl_x = 0;
			//if(s_gl_y<0) s_gl_y = 0;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if((s_gl_x < 0) || (s_gl_y < 0))  val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}
		// 2-2 top
		if(t_lc_y<r)
		{
			s_lc_x = t_lc_x + r, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x, s_gl_y = t_gl_y - r;
			//if(s_gl_y<0) s_gl_y = 0;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];		
			if(s_gl_y<0)	val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}
		// 2-3 top-right
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x + r + wg_x, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x + wg_x, s_gl_y = t_gl_y - r;
			//if(s_gl_x>=w) s_gl_x = w-1;
			//if(s_gl_y<0) s_gl_y = 0;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if((s_gl_x>=w)||(s_gl_y<0))	val = 0;

			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}
		// 2-4 bottom-left

		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x, s_lc_y = t_lc_y + r + wg_y;
			s_gl_x = t_gl_x - r, s_gl_y = t_gl_y + wg_y;
			//if(s_gl_x<0) s_gl_x = 0;

			//if(s_gl_y>=h) s_gl_y = h-1;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if((s_gl_x<0)||(s_gl_y>=h)) val = 0;
			else val = in[s_gl_y * w + s_gl_x];

			cache[s_lc_y * length_x + s_lc_x] = val;
		}	
		// 2-5 bottom
		if(t_lc_y<r)

		{
			s_lc_x = t_lc_x + r, s_lc_y = t_lc_y + r + wg_y;
			s_gl_x = t_gl_x, s_gl_y = t_gl_y + wg_y;
			//if(s_gl_y>=h) s_gl_y = h-1;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if(s_gl_y>=h) val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}	

		// 2-6 bottom-right
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x + r + wg_x, s_lc_y = t_lc_y + r + wg_y;

			s_gl_x = t_gl_x + wg_x, s_gl_y = t_gl_y + wg_y;
			//if(s_gl_x>=w) s_gl_x = w-1;
			//if(s_gl_y>=h) s_gl_y = h-1;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];

			if((s_gl_x>=w)||(s_gl_y>=h)) val = 0;
			else val = in[s_gl_y * w + s_gl_x];
			cache[s_lc_y * length_x + s_lc_x] = val;
		}	*/
		// 2-7 left

		if(t_lc_x<r)
		{
			//s_lc_x = t_lc_x, s_lc_y = t_lc_y + r;
			s_lc_x = t_lc_x;
			//s_gl_x = t_gl_x - r, s_gl_y = t_gl_y;
			s_gl_x = t_gl_x - r;

			//if(s_gl_x<0) s_gl_x = 0;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
			if(s_gl_x<0) val = 0;
			else val = in[s_gl_x];

			cache[s_lc_x] = val;
		}	
		// 2-8 right
		if(t_lc_x<r)
		{

			s_lc_x = t_lc_x + r + wg_x;
			s_gl_x = t_gl_x + wg_x;
			//if(s_gl_x>=w) s_gl_x = w-1;
			//cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];

			if(s_gl_x>=w) val = 0;
			else val = in[s_gl_x];
			cache[s_lc_x] = val;
		}

	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}

/*
	Optimized row transformation using local memory
*/
    __kernel void BoxRowsLmemOpt( __global const uchar4* uc4Source, __global unsigned int* uiDest,
                               __local uchar4* uc4LocalData,
                               unsigned int uiWidth, unsigned int uiHeight, int iRadius, 
                               float fScale)
    {
    	uint t_gl_x = get_global_id(0);
    	uint t_gl_y = get_global_id(1);    	
    
/*        // Compute x and y pixel coordinates from group ID and local ID indexes
        int globalPosX = ((int)get_group_id(0) * uiNumOutputPix) + (int)get_local_id(0) - iRadiusAligned;
        int globalPosY = (int)get_group_id(1);
        int iGlobalOffset = globalPosY * uiWidth + globalPosX;

        // Read global data into LMEM
        if (globalPosX >= 0 && globalPosX < uiWidth)
        {
            uc4LocalData[get_local_id(0)] = uc4Source[iGlobalOffset];
        }
        else 
        {
            uc4LocalData[get_local_id(0)].xyzw = (uchar4)0; 
        }

        // Synchronize the read into LMEM
        barrier(CLK_LOCAL_MEM_FENCE);*/
        
        // read data into local memory
        _clMemWriteBRO_FCTH1d(uc4Source, uiWidth, uiHeight, 0, iRadius, uc4LocalData, 0);

        // Compute (if pixel plus apron is within bounds)
        //if((globalPosX >= 0) && (globalPosX < uiWidth) && (get_local_id(0) >= iRadiusAligned) && (get_local_id(0) < (iRadiusAligned + (int)uiNumOutputPix)))
        //{
            // Init summation registers to zero
            float4 f4Sum = (float4)0.0f;

            // Do summation, using inline function to break up uint value from LMEM into independent RGBA values
            //int iOffsetX = (int)get_local_id(0) - iRadius;
            int iOffsetX = (int)get_local_id(0);
            int iLimit = iOffsetX + (2 * iRadius) + 1;
            for(iOffsetX; iOffsetX < iLimit; iOffsetX++)
            {
                f4Sum.x += uc4LocalData[iOffsetX].x;
                f4Sum.y += uc4LocalData[iOffsetX].y;
                f4Sum.z += uc4LocalData[iOffsetX].z;
                f4Sum.w += uc4LocalData[iOffsetX].w; 
            }

            // Use inline function to scale and convert registers to packed RGBA values in a uchar4, and write back out to GMEM
            //uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
            //uiDest[t_gl_y * uiWidth + t_gl_x] = rgbaFloat4ToUint(f4Sum, fScale);
            //uiDest[0] = rgbaFloat4ToUint(f4Sum, fScale);
            uiDest[t_gl_y * uiWidth + t_gl_x] = rgbaFloat4ToUint(f4Sum, fScale);
            
        //}
    }
