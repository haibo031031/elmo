void _clMemWriteBRO_FCTH(__global float * in, uint w, uint h, uint base_addr, int r, __local float * cache, uint p)
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
	uint s_gl_x = t_lc_x, s_gl_y = t_lc_y;	
	cache[s_lc_y * length_x + s_lc_x] = in[base_addr + s_gl_y * w + s_gl_x];
	if(r>0)
	{
		// 2-1 top-left		
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
