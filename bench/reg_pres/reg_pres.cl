#define BOX 16
__kernel void reg_pres(__global float *raw, const int rdim, const int cdim)
{	
	int idx;
	int i = get_global_id(0);
	int hist[BOX];
	if(i<rdim)
	{
		for(int k=0; k<BOX; k++)
		{
			hist[k] = 0;
		}
		
		for(int k=0; k<cdim; k++)
		{
			int idx = raw[i * cdim + k];
			hist[idx]++;
		}
		raw[i * cdim + 0] = hist[0];
	}
}

#define WG 64
__kernel void reg_pres_lm(__global float *raw, const int rdim, const int cdim, __local int* hist)
{	
	int idx;
	int i = get_global_id(0);
	int x_local = get_local_id(0);
	if(i<rdim)
	{
		for(int k=0; k<BOX; k++)
		{
			hist[x_local * BOX + k] = 0;
		}

		for(int k=0; k<cdim; k++)
		{
			int idx = raw[i * cdim + k];
			hist[x_local * BOX + idx]++;
		}
		raw[i * cdim + 0] = hist[x_local * BOX + 0];
	}
}

__kernel void reg_pres_gm(__global float *raw, const int rdim, const int cdim, __global int* hist)
{	
	int idx;
	int i = get_global_id(0);
	if(i<rdim)
	{
		for(int k=0; k<BOX; k++)
		{
			hist[cdim * i * BOX + k] = 0;
		}

		for(int k=0; k<cdim; k++)
		{
			int idx = raw[i * cdim + k];
			hist[cdim * i * BOX + idx]++;
		}
		raw[i * cdim + 0] = hist[cdim * i * BOX + 0];
	}
}

