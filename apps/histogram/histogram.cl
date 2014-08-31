#define LINEAR_MEM_ACCESS
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable 

/*
	with bank-conflict -- column-major 
*/
__kernel void _clLMSetCMI(__local uchar * cacher, uint bins, uchar val)
{	
	uint t_l_idx = get_local_id(0);
	for(uint i=0; i<bins; i++)
	{
		uint idx = t_l_idx * bins + i;
		cacher[idx] = val;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}

/*
	conflict-free -- row-major
*/
__kernel void _clLMSetRMI(__local uchar * cacher, uint bins, uchar val)
{	
	uint t_l_idx = get_local_id(0);
	uint tile = get_local_size(0);	
	uint size = bins * tile;
	for(uint i=0; i<size; i=i+tile)
	{
		uint idx = i + t_l_idx;
		cacher[idx] = val;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}


/**
 * @brief   do reduction operations on the local memory (Blocked)
 * @param   the local memory
 * @param   the number of bins/buckets
 * @param   ops: (0) addition; (1) max; (2) min; (4) others
 */
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

/**
 * @brief   do reduction operations on the local memory (Cyclic)
 * @param   the local memory
 * @param   the number of elements
 * @param   the number of bins/buckets
 * @param   ops: (0) addition; (1) max; (2) min; (4) others
 */
void _clLMReductionCyclic(__local uchar * lm, __global uint * res, uint bins, uint ops)
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
        uint y_idx = i * tile + t_l_idx;
        if(y_idx<bins)
        {
	        for(uint j = 0; j < tile; ++j)
    	    {
	            uint d_idx = y_idx * tile + j;
    	        //uint d_idx = (i * tile + t_l_idx) * tile + (j + t_l_idx%32)%tile;
    	        //uint d_idx = (i * tile + t_l_idx) * tile + ((j + 4 * (t_l_idx & 31)) & (tile-1));
    	        binCount += lm[d_idx];
   			}
   			res[i * tile + t_l_idx] = binCount;
   		}
    }
	return ;
}

/**
 * @brief   increase the value at pos
 * @param   the local memory
 * @param   the position of data element to be updated
*/
void _clLMInc(__local uchar * lm, uint pos)
{
	lm[pos]++;
	return ;
}

/**
	naive impl.: CMI + Cyclic
 */
__kernel
void histogram_1(__global const uint* data,
                  __local uchar* sharedArray,
                  __global uint* binResult,
                  uint bins)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);

    /* initialize shared array to zero */
	//_clLMSet_1(sharedArray, bins*groupSize, 0);
	_clLMSetCMI(sharedArray, bins, 0);
    
    /* calculate thread-histograms */
    for(int i = 0; i < bins; ++i)
    {
#ifdef LINEAR_MEM_ACCESS
        uint value = data[groupId * groupSize * bins + i * groupSize + localId];
#else
        uint value = data[globalId * bins + i];
#endif // LINEAR_MEM_ACCESS
        //sharedArray[localId * BIN_SIZE + value]++;
        //_clLMInc(sharedArray, localId * bins + value);
        _clLMInc(sharedArray, value * groupSize + localId);
    }      
    
    /* merge all thread-histograms into block-histogram */
    _clLMReductionCyclic(sharedArray, &(binResult[groupId * bins]), bins, 0);
}

/**	opt1 impl.: RMI + Cyclic
 * @brief   Calculates block-histogram bin whose bin size is 256
 * @param   data  input data pointer
 * @param   sharedArray shared array for thread-histogram bins
 * @param   binResult block-histogram array
 */
__kernel
void histogram_2(__global const uint* data,
                  __local uchar* sharedArray,
                  __global uint* binResult,
                  uint bins)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);

    /* initialize shared array to zero */
	//_clLMSet_2(sharedArray, bins*groupSize, 0);
	_clLMSetRMI(sharedArray, bins, 0);
    
    /* calculate thread-histograms */
    for(int i = 0; i < bins; ++i)
    {
#ifdef LINEAR_MEM_ACCESS
        uint value = data[groupId * groupSize * bins + i * groupSize + localId];
#else
        uint value = data[globalId * bins + i];
#endif // LINEAR_MEM_ACCESS
        //sharedArray[localId * BIN_SIZE + value]++;
        //_clLMInc(sharedArray, localId * BIN_SIZE + value);
        _clLMInc(sharedArray, value * groupSize + localId);

    }      
    
    /* merge all thread-histograms into block-histogram */
    //_clLMReduction(sharedArray, &(binResult[groupId * BIN_SIZE]), BIN_SIZE * groupSize, BIN_SIZE, 0);
    _clLMReductionCyclic(sharedArray, &(binResult[groupId * bins]), bins, 0);
}

/**
	opt2 impl.: CMI + blocked
 */
__kernel
void histogram_3(__global const uint* data,
                  __local uchar* sharedArray,
                  __global uint* binResult,
                  uint bins)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);

    /* initialize shared array to zero */
	//_clLMSet_1(sharedArray, bins*groupSize, 0);
	_clLMSetCMI(sharedArray, bins, 0);
    
    /* calculate thread-histograms */
    for(int i = 0; i < bins; ++i)
    {
#ifdef LINEAR_MEM_ACCESS
        uint value = data[groupId * groupSize * bins + i * groupSize + localId];
#else
        uint value = data[globalId * bins + i];
#endif // LINEAR_MEM_ACCESS
        //sharedArray[localId * BIN_SIZE + value]++;
        _clLMInc(sharedArray, localId * bins + value); // Blocked
        //_clLMInc(sharedArray, value * groupSize + localId); // Cyclic
    }      
    
    /* merge all thread-histograms into block-histogram */
    _clLMReductionBlocked(sharedArray, &(binResult[groupId * bins]), bins, 0);
}

/**
	opt3 impl.: RMI + blocked
 */
__kernel
void histogram_4(__global const uint* data,
                  __local uchar* sharedArray,
                  __global uint* binResult,
                  uint bins)
{
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);

    /* initialize shared array to zero */
	//_clLMSet_1(sharedArray, bins*groupSize, 0);
	_clLMSetRMI(sharedArray, bins, 0);
    
    /* calculate thread-histograms */
    for(int i = 0; i < bins; ++i)
    {
#ifdef LINEAR_MEM_ACCESS
        uint value = data[groupId * groupSize * bins + i * groupSize + localId];
#else
        uint value = data[globalId * bins + i];
#endif // LINEAR_MEM_ACCESS
        //sharedArray[localId * BIN_SIZE + value]++;
        _clLMInc(sharedArray, localId * bins + value); // Blocked
        //_clLMInc(sharedArray, value * groupSize + localId); // Cyclic
    }      
    
    /* merge all thread-histograms into block-histogram */
    _clLMReductionBlocked(sharedArray, &(binResult[groupId * bins]), bins, 0);
}
