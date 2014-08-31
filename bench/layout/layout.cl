/*
	partial reduction in the blocked way
*/
__kernel void layout_blocked(__global uint * out, __local uchar * lm, uint bins)
{
	uint t_lc_idx = get_local_id(0);
	uint tile = get_local_size(0);
	uint grp_idx = get_group_id(0);
	
	uint passes = 1;
	if(bins<tile) passes = 1;
	else passes = bins/tile;
	
	for(uint i = 0; i < passes; ++i)
    {
        uchar binCount = 0;
        uint x_idx = i * tile + t_lc_idx;
        if(x_idx<bins)
        {
	        for(uint j = 0; j < tile; ++j)
    	    {
    	    	uint d_idx = j * bins + x_idx;
    	    	//uint d_idx = ((j + t_l_idx%4)%tile) * numBins + (i * tile + t_l_idx);
    	    	//uint d_idx = ((j + (t_l_idx & 3)) & (tile-1)) * numBins + (i * tile + t_l_idx);
    	    	binCount += lm[d_idx];
    	    }
    	    out[grp_idx * bins + i * tile + t_lc_idx] = binCount;
    	}
    }
	return ;
}


/*
	partial reduction in the cyclic way
*/
__kernel void layout_cyclic(__global uint * out, __local uchar * lm, uint bins)
{
	uint t_lc_idx = get_local_id(0);
	uint tile = get_local_size(0);
	uint grp_idx = get_group_id(0);
	
	uint passes = 1;
	if(bins<tile) passes = 1;
	else passes = bins/tile;
	
	for(uint i = 0; i < passes; ++i)
    {
        uchar binCount = 0;
        uint y_idx = i * tile + t_lc_idx;
        if(y_idx<bins)
        {
	        for(uint j = 0; j < tile; ++j)
    	    {
    	        uint d_idx = y_idx * tile + j;
    	        //uint d_idx = (i * tile + t_l_idx) * tile + (j + t_l_idx%32)%tile;
    	        //uint d_idx = (i * tile + t_l_idx) * tile + ((j + 4 * (t_l_idx & 31)) & (tile-1));
    	        binCount += lm[d_idx];
   			}
   			out[grp_idx * bins + i * tile + t_lc_idx] = binCount;
   		}
    }
	return ;
}

/*
	partial reduction in the cyclic way
*/
__kernel void layout_cyclic_2(__global uint * out, __local uchar * lm, uint bins)
{
	uint t_lc_idx = get_local_id(0);
	uint tile = get_local_size(0);
	uint grp_idx = get_group_id(0);
	
	uint passes = 1;
	if(bins<tile) passes = 1;
	else passes = bins/tile;
	
	for(uint i = 0; i < passes; ++i)
    {
        uchar binCount = 0;
        uint y_idx = i * tile + t_lc_idx;
        if(y_idx<bins)
        {
	        for(uint j = 0; j < tile; ++j)
    	    {
    	        //uint d_idx = y_idx * tile + j;
    	        //uint d_idx = (i * tile + t_l_idx) * tile + (j + t_l_idx%32)%tile;
    	        //uint d_idx = (i * tile + t_lc_idx) * tile + ((j + (t_lc_idx & 31)) & (tile-1));
    	        uint d_idx = y_idx * tile + ((j + (t_lc_idx & 31)) & (tile-1));
    	        binCount += lm[d_idx];
   			}
   			out[grp_idx * bins + i * tile + t_lc_idx] = binCount;
   		}
    }
	return ;
}
