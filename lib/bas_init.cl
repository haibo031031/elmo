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
