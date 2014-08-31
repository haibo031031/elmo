void _clLMReductionBlocked(__local uchar * lm, __global uint * res, uint bins, uint ops)
{
	uint t_l_idx = get_local_id(0);
	uint tile = get_local_size(0);
	
	barrier(CLK_LOCAL_MEM_FENCE); 
	
	uint passes = 1;
	if(bins<tile) passes = 1;
	else passes = bins/tile;	
	for(uint i = 0; i < passes; ++i)
    {
        uint binCount = 0;
        uint x_idx = i * tile + t_l_idx;
        if(x_idx<bins)
        {
	        for(uint j = 0; j < tile; ++j)
    	    {
    	    	uint d_idx = j * bins + x_idx;
    	    	//uint d_idx = ((j + t_l_idx%4)%tile) * numBins + (i * tile + t_l_idx);
    	    	//uint d_idx = ((j + (t_l_idx & 3)) & (tile-1)) * numBins + (i * tile + t_l_idx);
    	    	binCount += lm[d_idx];
    	    }    	    
    	    res[i * tile + t_l_idx] = binCount;
		}
    }
	return ;
}
