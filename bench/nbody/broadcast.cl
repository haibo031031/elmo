
//__kernel void broadcast(volatile __global float *raw, __global float * out, const int size)
__kernel void broadcast(__global float * raw, __global float * out, const int size)
{	
	int idx = get_global_id(0);
	if(idx<size)
	{
		float curCost = 0.0f, deltaCost = 0.0f;
		curCost = raw[idx];
		deltaCost = 0.0f;
		for(int i=0; i<size; i++)
		{
			float tCost = raw[i];
			deltaCost += curCost - tCost;
		}
		out[idx] = deltaCost;
	}
}

//__kernel void broadcast_lm(volatile __global float * raw, __global float * out, const int size, __local float * cacher)
__kernel void broadcast_lm(__global float * raw, __global float * out, const int size, __local float * cacher)
{
	int t_g_idx = get_global_id(0);
	int t_l_idx = get_local_id(0);
	int tile = get_local_size(0);

	if(t_g_idx<size)
	{
		float curCost = 0.0f, deltaCost = 0.0f;
		curCost = raw[t_g_idx];
		deltaCost = 0.0f;
		for(int i=0; i<size; i=i+tile)
		{
			// load data
			cacher[t_l_idx] = raw[i + t_l_idx];
			barrier(CLK_LOCAL_MEM_FENCE);
			// use data
			for(int _i=0; _i<tile; _i++)
			{
				float tCost = cacher[_i];
				deltaCost += curCost - tCost;				
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		out[t_g_idx] = deltaCost;
	}		
}

