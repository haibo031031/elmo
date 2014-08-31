//__kernel void row_shared(__global float * raw, __global float * out, const int cdim, const int rdim)
__kernel void row_shared(const __global float *raw, __global float * out, const int cdim, const int rdim)
{	
	int x = get_global_id(0);
	int y = get_global_id(1);
	
	if(x<cdim && y<rdim)
	{
		int idx = y * cdim + x;		
		float sumCost = 0.0f;
		for(int i=0; i<cdim; i++)
		{
			float tCost = raw[y * cdim + i];
			sumCost += tCost;
		}
		out[idx] = sumCost;
	}
}
//__kernel void row_shared_lm(__global float * raw, __global float * out, const int cdim, const int rdim, __local float * cacher)
__kernel void row_shared_lm(const __global float * raw, __global float * out, const int cdim, const int rdim, __local float * cacher)
{
	int t_g_x = get_global_id(0);
	int t_g_y = get_global_id(1);
	int t_l_x = get_local_id(0);
	int t_l_y = get_local_id(1);
	int tile_x = get_local_size(0);
	int tile_y = get_local_size(1);
	
	
	if(t_g_x<cdim && t_g_y<rdim)
	{
		int idx = t_g_y * cdim + t_g_x;		
		float sumCost = 0.0f;
		for(int i=0; i<cdim; i=i+tile_x)
		{
			cacher[t_l_y * tile_x + t_l_x] = raw[t_g_y * cdim + i + t_l_x];
			barrier(CLK_LOCAL_MEM_FENCE);
			for(int _i=0; _i<tile_x; _i++)
			{
				float tCost = cacher[t_l_y * tile_x + _i];
				sumCost += tCost;				
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		out[idx] = sumCost;
	}
}

