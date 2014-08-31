void _clMemWriteBRO(__global uint * gl_data, uint cdim, uint rdim, int radius, __local uint * cacher)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	
	uint length_x = wg_x + 2 * radius;
	uint length_y = wg_y + 2 * radius;
	int ai = 0;
	int aj = 0;
	for(int j=0; j<length_y; j=j+wg_y)
	{
		aj = j + t_lc_y;
		if(aj<length_y)
		{
			for(int i=0; i<length_x; i=i+wg_x)
			{
				ai = i + t_lc_x;
				if(ai<length_x)
				{
					int id_local = (j + t_lc_y) * length_x + (i + t_lc_x);
					int x_tmp = (t_gl_x - radius) + i;
					if(x_tmp<0)	x_tmp = 0;
					else if(x_tmp>=cdim) x_tmp = cdim-1;
					int y_tmp = (t_gl_y - radius) + j;
					if(y_tmp<0)	y_tmp = 0;
					else if(y_tmp>=rdim) y_tmp = rdim-1;
					int id_global = y_tmp * cdim + x_tmp;
					cacher[id_local] = gl_data[id_global];
				}
			}
		}			
	}	
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}
