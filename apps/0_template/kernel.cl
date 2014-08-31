
/*
	subroutine: load data from global memory into local memory (TBT)
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
void _clMemWriteBRO_FCTH(__global uint * in, uint w, uint h, int r, __local uint * cache, uint p)
{
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
 * @brief   matrix transpose -- naive implementation
 * @param   input
 * @param   output
 * @param   wData: the width of input data array
 * @param   hData: the height of input data array
 */
__kernel void 
	matrix_transpose_naive(
		const __global  uint * in, \
		__global  uint * out, \
		uint wData, uint hData)
{	
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	
	if(t_gl_x<wData && t_gl_y<hData)
	{
		uint d_gl_x = t_gl_y;
		uint d_gl_y = t_gl_x;
		uint d_gl_idx = d_gl_y * wData + d_gl_x;
		uint val = in[d_gl_idx];
		//out[t_gl_y * wData + t_gl_x] = val;
		out[t_gl_y * hData + t_gl_x] = val;		
	}	
}

/**
 * @brief   matrix transpose -- with local memory (FCTH)
 * @param   input
 * @param   output
 * @param	the local memory space
 * @param   wData: the width of input data array
 * @param   hData: the height of input data array
 */
__kernel void 
	matrix_transpose_lc(
		const __global  uint * in, \
		__global  uint * out, \
		__local uint * cacher, \
		uint wData, uint hData)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	uint grp_x = get_group_id(0);
	uint grp_y = get_group_id(1);
	uint length_x = wg_x;
	
	// load data into local memory
	_clMemWriteBRO_FCTH(in, wData, hData, 0, cacher, 0);
	
	// perform computation using the data in the LM
	if(t_gl_x<wData && t_gl_y<hData)
	{	
		//uint d_gl_x = t_gl_y; // transform within one work-group
		uint d_lc_x = t_lc_y;
		//uint d_gl_y = t_gl_x;
		uint d_lc_y = t_lc_x;
		//uint d_gl_idx = d_gl_y * wData + d_gl_x;
		uint d_lc_idx = d_lc_y * length_x + d_lc_x;
		//uint val = in[d_gl_idx];
		uint val = cacher[d_lc_idx];
		//out[t_gl_y * wData + t_gl_x] = val;
		//out[t_gl_y * hData + t_gl_x] = val;
		uint d_gl_x = wg_y * grp_y + t_lc_x; // transform the work-groups
		uint d_gl_y = wg_x * grp_x + t_lc_y;
		out[d_gl_y * hData + d_gl_x] = val;
	}	
    
	return ;
}

/**

 * @brief   matrix transpose -- with local memory (FCTH) bank-conflicts removal
 * @param   input
 * @param   output
 * @param	the local memory space
 * @param   wData: the width of input data array
 * @param   hData: the height of input data array
 */
__kernel void 
	matrix_transpose_lc_bcr(
		const __global  uint * in, \
		__global  uint * out, \
		__local uint * cacher, \
		uint wData, uint hData)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	uint grp_x = get_group_id(0);
	uint grp_y = get_group_id(1);
	uint length_x = wg_x + 1;
	
	// load data into local memory
	_clMemWriteBRO_FCTH(in, wData, hData, 0, cacher, 1);
	
	// perform computation using the data in the LM
	if(t_gl_x<wData && t_gl_y<hData)
	{	
		//uint d_gl_x = t_gl_y; // transform within one work-group
		uint d_lc_x = t_lc_y;
		//uint d_gl_y = t_gl_x;
		uint d_lc_y = t_lc_x;
		//uint d_gl_idx = d_gl_y * wData + d_gl_x;
		uint d_lc_idx = d_lc_y * length_x + d_lc_x;
		//uint val = in[d_gl_idx];
		uint val = cacher[d_lc_idx];
		//out[t_gl_y * wData + t_gl_x] = val;
		//out[t_gl_y * hData + t_gl_x] = val;
		uint d_gl_x = wg_y * grp_y + t_lc_x; // transform the work-groups
		uint d_gl_y = wg_x * grp_x + t_lc_y;
		out[d_gl_y * hData + d_gl_x] = val;
	}	
    
	return ;
}
