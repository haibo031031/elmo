/*
	redution -- native version (27/08/2012)
*/
__kernel void reduction(__global uint * input, __global uint * output, uint size)
{
	uint gl_idx = get_global_id(0);
	uint lc_idx = get_local_id(0);
	uint wg_size = get_local_size(0);
	uint grp_idx = get_group_id(0);
	uint grp_start = wg_size * grp_idx;
	
	input[gl_idx] += input[gl_idx + size/2];
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	// do reduction within one work-group
    for(uint s = wg_size >> 1; s > 0; s >>= 1) 
    {
        if(lc_idx < s) 
        {
            input[gl_idx] += input[gl_idx + s];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if(lc_idx == 0) output[grp_idx] = input[grp_start];	
    
    // do reduction accross work-groups    
}

__kernel void reduction_native(__global uint * input, __global uint * output, uint rake_size, uint N)
{
	uint gl_idx = get_global_id(0);
	uint lc_idx = get_local_id(0);
	uint wg_size = get_local_size(0);
	uint grp_idx = get_group_id(0);
	uint grp_start = wg_size * grp_idx;
		
	// load data from global memory to local memory
	for(uint i=0; i<N; i++)
	{
		//lm[lc_idx] += input[i * rake_size + gl_idx]; //reduction(+)
		input[gl_idx] += input[i * rake_size + gl_idx];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	// do reduction within one work-group on the local memory
    for(uint s = wg_size >> 1; s > 0; s >>= 1) 
    {
        if(lc_idx < s) 
        {
            //lm[lc_idx] += lm[lc_idx + s];
            //input[grp_start + lc_idx] += input[grp_start + lc_idx + s];
            input[gl_idx] += input[gl_idx + s];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // write result for this block to global mem
    //if(lc_idx == 0) output[grp_idx] = lm[0];
    if(lc_idx == 0) output[grp_idx] = input[grp_start];	
}

/*
	size: the total number of threads;
	N:	the number of data elements processed by one thread
*/
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
