/*
	subroutine: load data from global memory into local memory
*/
void _clMemWriteBRO(__global uint * gl_data, uint cdim, uint rdim, uint radius, __local uint * cacher, uint length_x, uint length_y)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);

	uint ai = 0;
	uint aj = 0;
	for(uint j=0; j<length_y; j=j+wg_y)
	{
		aj = j + t_lc_y;
		if(aj<length_y)
		{
			for(uint i=0; i<length_x; i=i+wg_x)
			{
				ai = i + t_lc_x;
				if(ai<length_x)
				{
					uint id_local = (j + t_lc_y) * length_x + (i + t_lc_x);
					int x_tmp = (t_gl_x - radius) + i;
					if(x_tmp<0)	x_tmp = 0;
					else if(x_tmp>=cdim) x_tmp = cdim-1;
					int y_tmp = (t_gl_y - radius) + j;
					if(y_tmp<0)	y_tmp = 0;
					else if(y_tmp>=rdim) y_tmp = rdim-1;
					uint id_global = y_tmp * cdim + x_tmp;
					cacher[id_local] = gl_data[id_global];
				}
			}
		}			
	}	
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}
/*
	load data in a tile-by-tile mode
*/
__kernel void g2l_TBT(__global uint * in, __global uint * out, __local uint * cache, uint w, uint h, uint r, \
					uint length_x, uint length_y)
{	
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_gl_idx = t_gl_y * w + t_gl_x;
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	uint t_lc_idx = t_lc_y * wg_x + t_lc_x;
	
	uint val = 0;	
	// load data from global memory to local memory
	_clMemWriteBRO(in, w, h, r, cache, length_x, length_y);
	
	// write the data (index 0) back to the output
	val = cache[t_lc_idx];
	out[t_gl_idx] = val;
}

/*
	load data First Central Then Halo (FCTH)
*/
__kernel void g2l_FCTH(__global uint * in, __global uint * out, __local uint * cache, uint w, uint h, uint r, \
					uint length_x, uint length_y)
{	
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_gl_idx = t_gl_y * w + t_gl_x;
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	uint t_lc_idx = t_lc_y * wg_x + t_lc_x;
	
	uint val = 0;	
	// load data from global memory to local memory
	// 1. central part
	uint s_lc_x = t_lc_x + r;
	uint s_lc_y = t_lc_y + r;
	cache[s_lc_y * length_x + s_lc_x] = in[t_gl_idx];
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
	barrier(CLK_LOCAL_MEM_FENCE);
	// write the data (index 0?) back to the output
	val = cache[t_lc_idx];
	out[t_gl_idx] = val;
}
