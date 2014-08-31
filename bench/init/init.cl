/*
	with bank-conflict -- column-major 
*/
__kernel void _clLMSetCMI(__local float * cacher, const int bins)
{	
	int t_l_idx = get_local_id(0);
	for(int i=0; i<bins; i++)
	{
		int idx = t_l_idx * bins + i;
		cacher[idx] = t_l_idx;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}

/*
	conflict-free -- row-major
*/
__kernel void _clLMSetRMI(__local float * cacher, const int bins)
{	
	int t_l_idx = get_local_id(0);
	int tile = get_local_size(0);	
	int size = bins * tile;
	for(int i=0; i<size; i=i+tile)
	{
		int idx = i + t_l_idx;
		cacher[idx] = t_l_idx;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}
