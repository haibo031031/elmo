__kernel void reduction_N(__global uint * input, __global uint * output, __local uint * lm, uint rake_size, uint N)
{
	uint gl_idx = get_global_id(0);
	uint lc_idx = get_local_id(0);
	uint wg_size = get_local_size(0);
	uint grp_idx = get_group_id(0);
	
	// load data from global memory to local memory
	for(uint i=0; i<N; i++)
	{
		lm[lc_idx] += input[i * rake_size + gl_idx]; //reduction(+)
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// do reduction within one work-group on the local memory
    for(uint s = wg_size >> 1; s > 0; s >>= 1) 
    {
        if(lc_idx < s) 
        {
            lm[lc_idx] += lm[lc_idx + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(lc_idx == 0) output[grp_idx] = lm[0];	
}
