//////////////////////////////////////////////////////////////////////
//Name: ConstantBlockAggregation.cl 
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: aggregating cost on a constant block - opencl kernel
///////////////////////////////////////////////////////////////////////

/*
	revised by Jianbin on 27/02/2012 on
	+ storage format and coalesced memory access
	+ using local memory (28/02 dynamic allocation)
	+ 
*/


/*__kernel void boxAggregate(__global float *raw_cost, __global float * ref_cost, __global float * tar_cost, \
	const int rdim, const int cdim, const int dispRange, const int wndwRad, \
	__local float * cache, const int tile_size, const int length){
	int grp_c = get_group_id(0);
	int grp_r = get_group_id(1);
	int s_c = 16;
	int s_r = 16;
	int l_c = get_local_id(0);
	int l_r = get_local_id(1);	
	int x_idx = grp_c * s_c + l_c;
	int y_idx = grp_r * s_r + l_r;
	//__local float cache[TILE_SIZE + B_SIZE * 2][TILE_SIZE + B_SIZE * 2];

	int wnd_side = 2 * wndwRad + 1;
	int wndwSize = wnd_side * wnd_side;	
		
	
	for(int d = 0; d < dispRange; d++){
	
		// + load data into local memory
		int x = x_idx - wndwRad;
		int y = y_idx - wndwRad;
		
		if(x >= cdim) x -= cdim;
		else if(x < 0) x += cdim;
		if(y >= rdim) y -= rdim;
		else if(y < 0) y += rdim;
		cache[l_r * length + l_c] = raw_cost[d * rdim * cdim + cdim * y + x];
		barrier(CLK_LOCAL_MEM_FENCE);
			
		// + use the shared memory and get the summary
		if(l_c<16 && l_r<16){
			float cost_aggr=0;
			for(int y = 0; y < wnd_side; y++){
				for(int x = 0; x < wnd_side; x++){
					int x_start = l_c + x;
					int y_start = l_r + y;
					cost_aggr += cache[y_start * length + x_start]; 
				}
			}
			cost_aggr = (float)(cost_aggr)/(float)(wndwSize);
			ref_cost[d * rdim * cdim + cdim * y_idx + x_idx] = cost_aggr;
			int tar_x = 0;
			if(x_idx-d<0) tar_x = x_idx - d + cdim;
			else tar_x = x_idx-d;
			tar_cost[d * rdim * cdim + cdim * y_idx + tar_x] = cost_aggr;
		}		
	}

}*/

__kernel void boxAggregate(__global float *raw_cost, __global float * ref_cost, \
	__global float * tar_cost,
	const int rdim, const int cdim, 
	const int dispRange, const int wndwRad){

	int x = get_global_id(0);
	int y = get_global_id(1);
	int wndwSize = (2*wndwRad+1)*(2*wndwRad+1);
	int ytar=0, xtar=0;
	float costAggr=0;
	if(x<cdim && y<rdim){
		for(int d=0; d<dispRange; d++){
			costAggr=0;
			for(int y_=-wndwRad; y_ <= wndwRad; y_++){
				ytar = y+y_;
				if(ytar>=rdim) ytar -= rdim;
				else if(ytar<0) ytar += rdim;
				for(int x_=-wndwRad; x_ <= wndwRad; x_++){
					xtar = x+x_;
					if(xtar>=cdim) xtar -= cdim;
					else if(xtar<0) xtar += cdim;
					//costAggr += raw_cost[d+dispRange*(xtar+cdim*ytar)];
					costAggr += raw_cost[d * cdim * rdim + (xtar+cdim*ytar)];  
				}
			}
			costAggr = (float)(costAggr)/(float)(wndwSize);
			//ref_cost[d + dispRange * (x + cdim * y)] = costAggr;
			ref_cost[d * cdim * rdim + (x + cdim * y)] = costAggr;
			int temp_x = 0;
			if(x-d<0) temp_x = x - d + cdim;
			else temp_x = x-d;
			//tar_cost[d+dispRange*(temp_x+cdim*y)] = costAggr;
			tar_cost[d * cdim * rdim + (temp_x+cdim*y)] = costAggr;
		}
	}	
}

__kernel void horzIntegral(__global float *raw_cost, \
		const int rdim, const int cdim, const int dispRange){
	
	int idx,idx_prev;
	int d = get_global_id(0);
	int y = get_global_id(1);
	if(y<rdim)
	{
		for(int x=1; x<cdim; x++)
		{
			idx = y*cdim+x;
			raw_cost[d * cdim * rdim + idx] += raw_cost[d * cdim * rdim + (idx-1)]; 
		}			
	}
}

__kernel void vertIntegral(__global float *raw_cost, \
		const int rdim, const int cdim, const int dispRange){
	
	int idx,idx_prev;
	int d = get_global_id(0);
	int x = get_global_id(1);
	if(x<cdim)
	{
		for(int y=1; y<rdim; y++)
		{
			idx = y*cdim+x;
			raw_cost[d * cdim * rdim + idx] += raw_cost[d * cdim * rdim + (idx-cdim)]; 
		}
	}
}

__kernel void boxAggregateIntegral(const __global float *integral_cost, __global float *ref_cost,\
		__global float *tar_cost, const int rdim, const int cdim, const int dispRange, const int wndwRad){
	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int wndwSize = (2*wndwRad+1)*(2*wndwRad+1);
	int ytar=0, xtar=0,x_tar,x_left,x_right,y_top,y_bottom,Idx;
	float costAggr=0, refCost=0,tarCost=0;
	if(x<cdim && y<rdim){		
		for(int d=0; d<dispRange; d++){
			int base = d * cdim * rdim;
			Idx = base + (y*cdim+x);
			x_left = x-wndwRad;
			if(x_left<0) x_left=0;
			x_right = x+wndwRad;
			if(x_right>=cdim) x_right = cdim-1;
			y_top = y-wndwRad;
			if(y_top<0) y_top=0;
			y_bottom = y+wndwRad;
			if(y_bottom>=rdim) y_bottom = rdim-1;
			costAggr=integral_cost[base + (y_bottom*cdim+x_right)]+
					integral_cost[base + (y_top*cdim+x_left)]-
					integral_cost[base + (y_bottom*cdim+x_left)]-
					integral_cost[base + (y_top*cdim+x_right)];
			ref_cost[Idx] = costAggr;

			int temp_x = 0;
			if(x-d<0) temp_x = x - d + cdim;
			else temp_x = x-d;
			tar_cost[base + (y*cdim+temp_x)] = costAggr;
		}
	}
}
