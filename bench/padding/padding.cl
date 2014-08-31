/*
	with bank-conflict -- (16x16)
*/
__kernel void _clLMNative(__global float * out, __local float * cacher, uint w)
{	
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_gl_idx = t_gl_y * w + t_gl_x;
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = 16;
	uint wg_y = 16;
	
	uint t_lc_idx = t_lc_y * wg_x + t_lc_x;
	cacher[t_lc_idx] = (float)t_lc_idx;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	uint t_lc_idx_trans = t_lc_x * wg_y + t_lc_y;
	//float val = cacher[t_lc_idx_trans];
	float val = (float)t_lc_idx;
	out[t_gl_idx] = val;	
	return ;
}

/*
	conflict-free -- (17x16)
*/
__kernel void _clLMOpt(__global float * out, __local float * cacher, uint w)
{	
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_gl_idx = t_gl_y * w + t_gl_x;
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = 17;
	uint wg_y = 16;
	
	uint t_lc_idx = t_lc_y * wg_x + t_lc_x;
	cacher[t_lc_idx] = (float)t_lc_idx;
	barrier(CLK_LOCAL_MEM_FENCE);
	
	uint t_lc_idx_trans = t_lc_x * wg_x + t_lc_y;
	//float val = cacher[t_lc_idx_trans];
	float val = (float)t_lc_idx;
	out[t_gl_idx] = val;	
	return ;
}

