/*
	subroutine: load data from global memory into local memory (TBT)
	@params gl_data: global data
	@params cdim: the collumn dimension
	@params rdim: the row dimension
	@params radius: the radius of data elements
	@params cacher: the local memory
*/
void _clMemWriteBRO(__global uint * gl_data, uint cdim, uint rdim, int radius, __local uint * cacher)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	
	uint length_x = wg_x + 2 * radius;
	uint length_y = wg_y + 2 * radius;
	int ai = 0;
	int aj = 0;
	for(int j=0; j<length_y; j=j+wg_y)
	{
		aj = j + t_lc_y;
		if(aj<length_y)
		{
			for(int i=0; i<length_x; i=i+wg_x)
			{
				ai = i + t_lc_x;
				if(ai<length_x)
				{
					int id_local = (j + t_lc_y) * length_x + (i + t_lc_x);
					int x_tmp = (t_gl_x - radius) + i;
					if(x_tmp<0)	x_tmp = 0;
					else if(x_tmp>=cdim) x_tmp = cdim-1;
					int y_tmp = (t_gl_y - radius) + j;
					if(y_tmp<0)	y_tmp = 0;
					else if(y_tmp>=rdim) y_tmp = rdim-1;
					int id_global = y_tmp * cdim + x_tmp;
					cacher[id_local] = gl_data[id_global];
				}
			}
		}			
	}	
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}

/*
	subroutine: load data from global memory into local memory (FCTH)
*/
void _clMemWriteBRO_FCTH(__global uint * in, uint w, uint h, int r, __local uint * cache)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_gl_idx = t_gl_y * w + t_gl_x;
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	
	uint length_x = wg_x + 2 * r;
	uint length_y = wg_y + 2 * r;
	// 1. central part
	uint s_lc_x = t_lc_x + r;
	uint s_lc_y = t_lc_y + r;
	cache[s_lc_y * length_x + s_lc_x] = in[t_gl_idx];
	if(r>0)
	{
		// 2-1 top-left
		int s_gl_x, s_gl_y;	
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x - r, s_gl_y = t_gl_y - r;
			if(s_gl_x<0) s_gl_x = 0;
			if(s_gl_y<0) s_gl_y = 0;
			cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
		}
		// 2-2 top
		if(t_lc_y<r)
		{
			s_lc_x = t_lc_x + r, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x, s_gl_y = t_gl_y - r;
			if(s_gl_y<0) s_gl_y = 0;
			cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];		
		}
		// 2-3 top-right
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x + r + wg_x, s_lc_y = t_lc_y;
			s_gl_x = t_gl_x + wg_x, s_gl_y = t_gl_y - r;
			if(s_gl_x>=w) s_gl_x = w-1;
			if(s_gl_y<0) s_gl_y = 0;
			cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
		}
		// 2-4 bottom-left
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x, s_lc_y = t_lc_y + r + wg_y;
			s_gl_x = t_gl_x - r, s_gl_y = t_gl_y + wg_y;
			if(s_gl_x<0) s_gl_x = 0;
			if(s_gl_y>=h) s_gl_y = h-1;
			cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
		}	
		// 2-5 bottom
		if(t_lc_y<r)
		{
			s_lc_x = t_lc_x + r, s_lc_y = t_lc_y + r + wg_y;
			s_gl_x = t_gl_x, s_gl_y = t_gl_y + wg_y;
			//if(s_gl_x<0) s_gl_x = 0;
			if(s_gl_y>=h) s_gl_y = h-1;
			cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
		}	
		// 2-6 bottom-right
		if(t_lc_x<r && t_lc_y<r)
		{
			s_lc_x = t_lc_x + r + wg_x, s_lc_y = t_lc_y + r + wg_y;
			s_gl_x = t_gl_x + wg_x, s_gl_y = t_gl_y + wg_y;
			if(s_gl_x>=w) s_gl_x = w-1;
			if(s_gl_y>=h) s_gl_y = h-1;
			cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
		}	
		// 2-7 left
		if(t_lc_x<r)
		{
			s_lc_x = t_lc_x, s_lc_y = t_lc_y + r;
			s_gl_x = t_gl_x - r, s_gl_y = t_gl_y;
			if(s_gl_x<0) s_gl_x = 0;
			//if(s_gl_y<0) s_gl_y = 0;
			cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
		}	
		// 2-8 right
		if(t_lc_x<r)
		{
			s_lc_x = t_lc_x + r + wg_x, s_lc_y = t_lc_y + r;
			s_gl_x = t_gl_x + wg_x, s_gl_y = t_gl_y;
			if(s_gl_x>=w) s_gl_x = w-1;
			//if(s_gl_y<0) s_gl_y = 0;
			cache[s_lc_y * length_x + s_lc_x] = in[s_gl_y * w + s_gl_x];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}


/**
 * @brief   image convolution -- native implementation
 * @param   input
 * @param   filter
 * @param   output
 * @param   wData: the width of input data array
 * @param	r:	the radius of filter 
 */
__kernel void 
	image_convolution_native(
		const __global  uint * in, \
		__constant uint * filter, \
		__global  uint * out, \
		uint wData, uint hData, int r)
{	
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	
	uint val = 0;	
	uint wF = 2 * r + 1;
	uint area = wF * wF;
    for(int _y=-r, __y=0; _y<=r; _y++, __y++)
    {
    	for(int _x=-r, __x=0; _x<=r; _x++, __x++)
    	{
    		int d_gl_x = t_gl_x + _x;
    		if(d_gl_x<0) d_gl_x = 0;
    		else if(d_gl_x>=wData) d_gl_x = wData - 1;    		
    		int d_gl_y = t_gl_y + _y;
    		if(d_gl_y<0) d_gl_y = 0;
    		else if(d_gl_y>=hData) d_gl_y = hData - 1;    		
    		int d_gl_in = d_gl_y * wData + d_gl_x;    		
    		int d_f_idx = __y * wF + __x;
    		
    		val = val + in[d_gl_in] * filter[d_f_idx];
    	}
    }
    val = val/area;
    uint d_gl_idx = t_gl_y * wData + t_gl_x;
    out[d_gl_idx] = val;
}

/**
 * @brief   image convolution -- with local memory
 * @param   input
 * @param   filter
 * @param   output
 * @param	the local memory space
 * @param   wData: the width of input data array
 * @param	r:	the radius of filter
 */
__kernel void 
	image_convolution_lc_tbt(
		const __global  uint * in, \
		__constant uint * filter, \
		__global  uint * out, \
		__local uint * cacher, \
		uint wData, uint hData, int r)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint length_x = wg_x + 2 * r;
	
	// load data into local memory
	_clMemWriteBRO(in, wData, hData, r, cacher);
	
	// perform computation using the data in the LM
	uint val = 0;	
	uint wF = 2 * r + 1;
	uint area = wF * wF;
    for(int _y=-r, __y=0; _y<=r; _y++, __y++)
    {
    	for(int _x=-r, __x=0; _x<=r; _x++, __x++)
    	{
    		//uint d_gl_x = t_gl_x + _x; 
    		int d_lc_x = t_lc_x + __x;
    		//if(d_gl_x<0) d_gl_x = 0;
    		//else if(d_gl_x>=wData) d_gl_x = wData - 1;    		
    		//uint d_gl_y = t_gl_y + _y;
    		int d_lc_y = t_lc_y + __y;
    		//if(d_gl_y<0) d_gl_y = 0;
    		//else if(d_gl_y>=hData) d_gl_y = hData - 1;    		
    		//uint d_gl_in = d_gl_y * wData + d_gl_x;    		
    		int d_lc_in = d_lc_y * length_x + d_lc_x;
    		int d_f_idx = __y * wF + __x;
    		
    		val = val + cacher[d_lc_in] * filter[d_f_idx];
    	}
    }
    val = val/area;
    uint d_gl_idx = t_gl_y * wData + t_gl_x;
    out[d_gl_idx] = val;	
    
	return ;
}

/**
 * @brief   image convolution -- with local memory
 * @param   input
 * @param   filter
 * @param   output
 * @param	the local memory space
 * @param   wData: the width of input data array
 * @param	r:	the radius of filter
 */
__kernel void 
	image_convolution_lc_fcth(
		const __global  uint * in, \
		__constant uint * filter, \
		__global  uint * out, \
		__local uint * cacher, \
		uint wData, uint hData, int r)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint length_x = wg_x + 2 * r;
	
	// load data into local memory
	_clMemWriteBRO_FCTH(in, wData, hData, r, cacher);
	
	// perform computation using the data in the LM
	uint val = 0;	
	uint wF = 2 * r + 1;
	uint area = wF * wF;
    for(int _y=-r, __y=0; _y<=r; _y++, __y++)
    {
    	for(int _x=-r, __x=0; _x<=r; _x++, __x++)
    	{
    		//uint d_gl_x = t_gl_x + _x; 
    		int d_lc_x = t_lc_x + __x;
    		//if(d_gl_x<0) d_gl_x = 0;
    		//else if(d_gl_x>=wData) d_gl_x = wData - 1;    		
    		//uint d_gl_y = t_gl_y + _y;
    		int d_lc_y = t_lc_y + __y;
    		//if(d_gl_y<0) d_gl_y = 0;
    		//else if(d_gl_y>=hData) d_gl_y = hData - 1;    		
    		//uint d_gl_in = d_gl_y * wData + d_gl_x;    		
    		int d_lc_in = d_lc_y * length_x + d_lc_x;
    		int d_f_idx = __y * wF + __x;
    		
    		val = val + cacher[d_lc_in] * filter[d_f_idx];
    	}
    }
    val = val/area;
    uint d_gl_idx = t_gl_y * wData + t_gl_x;
    out[d_gl_idx] = val;	
    
	return ;
}
