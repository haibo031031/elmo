
__kernel void broadcast(const __global float *raw, __global float * out, const int cdim, const int rdim)
//__kernel void broadcast(__global float * raw, __global float * out, const int cdim, const int rdim)
{	
	int x = get_global_id(0);
	int y = get_global_id(1);
	float curCost = 0.0f, deltaCost = 0.0f;
	if(x<cdim && y<rdim)
	{
		int idx = y * cdim + x;
		curCost = raw[idx];
		deltaCost = 0.0f;
		for(int j=0; j<rdim; j++)
		{
			for(int i=0; i<cdim; i++)
			{
				float tCost = raw[j * cdim + i];
				deltaCost += curCost - tCost;
			}
		}
		out[idx] = deltaCost;
	}
}

__kernel void broadcast_lm(const __global float * raw, __global float * out, const int cdim, const int rdim, __local float * cacher)
//__kernel void broadcast_lm(__global float * raw, __global float * out, const int cdim, const int rdim, __local float * cacher)
{
	int t_g_x = get_global_id(0);
	int t_g_y = get_global_id(1);
	int t_l_x = get_local_id(0);
	int t_l_y = get_local_id(1);
	int tile_x = get_local_size(0);
	int tile_y = get_local_size(1);
	
	if(t_g_x<cdim && t_g_y<rdim)
	{
		int d_g_idx = t_g_y * cdim + t_g_x;
		float curCost = 0.0f, deltaCost = 0.0f;
		curCost = raw[d_g_idx];
		
		//for(int _y=0; _y<rdim; _y++)
		for(int _y=0; _y<rdim; _y=_y+tile_y)
		{
			//for(int _x=0; _x<cdim; _x++)
			for(int _x=0; _x<cdim; _x=_x+tile_x)
			{
				//float tCost = raw[_y * cdim + _x];
				//int d_g_x = t_g_x + _x;
				//int d_g_y = t_g_y + _y;
				int d_l_x = t_l_x;
				int d_l_y = t_l_y;
				int d_g_x = _x + d_l_x;
				int d_g_y = _y + d_l_y;

				cacher[d_l_y * tile_x + d_l_x] = raw[d_g_y * cdim + d_g_x];
				barrier(CLK_LOCAL_MEM_FENCE);
				
				for(int y__=0; y__<tile_y; y__++)
				{
					for(int x__=0; x__<tile_x; x__++)
					{
						int _d_l_x = x__;
						int _d_l_y = y__;
						float tCost = cacher[_d_l_y * tile_x + _d_l_x];
						deltaCost += curCost - tCost;
					}
				}
				
				barrier(CLK_LOCAL_MEM_FENCE);				
			}
		}
		out[d_g_idx] = deltaCost;
	}		
}

