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

void _clWriteLM(__global float * gl_data, int cdim, int rdim, int offset_x, int offset_y, __local float * cacher, int length_x, int length_y, int d){

	int t_gl_x = get_global_id(0);
	int t_gl_y = get_global_id(1);
	int t_lc_x = get_local_id(0);
	int t_lc_y = get_local_id(1);
	int wg_x = get_local_size(0);
	int wg_y = get_local_size(1);

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
					int x_tmp = (t_gl_x - offset_x) + i;
					if(x_tmp<0)	x_tmp = 0;
					else if(x_tmp>=cdim) x_tmp = cdim-1;
					int y_tmp = (t_gl_y - offset_y) + j;
					if(y_tmp<0)	y_tmp = 0;
					else if(y_tmp>=rdim) y_tmp = rdim-1;
					int id_global = d * rdim * cdim + y_tmp * cdim + x_tmp;
					cacher[id_local] = gl_data[id_global];
				}
			}
		}			
	}	
	barrier(CLK_LOCAL_MEM_FENCE);
}

float _clReadLM(__local float * cacher, int length_x, int offset_x, int offset_y, int cur_offset_x, int cur_offset_y){

	int t_lc_x = get_local_id(0);
	int t_lc_y = get_local_id(1);
	int d_lc_x = t_lc_x + offset_x + cur_offset_x;
	int d_lc_y = t_lc_y + offset_y + cur_offset_y;

	return cacher[d_lc_y * length_x + d_lc_x];
		
}

__kernel void boxAggregate_lm(__global float *raw_cost, __global float * ref_cost, \
	__global float * tar_cost,
	const int rdim, const int cdim, 
	const int dispRange, const int wndwRad, \
	__local float * cacher, const int tile_size, const int length){

	int x = get_global_id(0);
	int y = get_global_id(1);
	int wndwSize = (2*wndwRad+1)*(2*wndwRad+1);
	int ytar=0, xtar=0;
	float costAggr=0;
	if(x<cdim && y<rdim){
		for(int d=0; d<dispRange; d++){
			_clWriteLM(raw_cost, cdim, rdim, wndwRad, wndwRad, cacher, length, length, d);
			costAggr=0;
			for(int y_=-wndwRad; y_ <= wndwRad; y_++){
				ytar = y+y_;
				if(ytar>=rdim) ytar -= rdim;
				else if(ytar<0) ytar += rdim;
				for(int x_=-wndwRad; x_ <= wndwRad; x_++){
					xtar = x+x_;
					if(xtar>=cdim) xtar -= cdim;
					else if(xtar<0) xtar += cdim;				
					//costAggr += raw_cost[d * cdim * rdim + (xtar+cdim*ytar)];  
					costAggr += _clReadLM(cacher, length, wndwRad, wndwRad, x_, y_);
				}
			}
			costAggr = (float)(costAggr)/(float)(wndwSize);
			ref_cost[d * cdim * rdim + (x + cdim * y)] = costAggr;	
		}
	}	
}

/*__kernel void boxAggregate_lm(__global float *raw_cost, __global float * ref_cost, __global float * tar_cost, \
	const int rdim, const int cdim, const int dispRange, const int wndwRad, \
	__local float * cache, const int tile_size, const int length){
	int x = get_global_id(0);
	int y = get_global_id(1);
	int x_local = get_local_id(0);
	int y_local = get_local_id(1);

	int wnd_side = 2 * wndwRad + 1;
	int wndwSize = wnd_side * wnd_side;	
		
	if(x<cdim && y<rdim){
		int iter = (length%tile_size==0)?(length/tile_size):(length/tile_size+1);
		for(int d = 0; d < dispRange; d++){
		
			int base = d * rdim * cdim;
			// + load data into cache
			int ai = 0;
			int aj = 0;
			for(int j=0; j<length; j=j+tile_size)	
			{
				aj = j + y_local;
				if(aj<length)
				{
					for(int i=0; i<length; i=i+tile_size)
					{
						ai = i + x_local;
						if(ai<length)
						{
							int id_local = (j + y_local) * length + (i + x_local);					
							int x_tmp = (x - tile_size) + i;
							if(x_tmp<0)	x_tmp = x_tmp + cdim;
							else if(x_tmp>=cdim) x_tmp = x_tmp - cdim;
							int y_tmp = (y - tile_size) + j;
							if(y_tmp<0)	y_tmp = y_tmp + rdim;
							else if(y_tmp>=rdim) y_tmp = y_tmp - rdim;
							int id_global = base + y_tmp * cdim + x_tmp;
							cache[id_local] = raw_cost[id_global];										
						}
					}
				}			
			}
			barrier(CLK_LOCAL_MEM_FENCE);
				
			// + use the shared memory and get the summary
			float cost_aggr=0;
			for(int y_ = 0; y_ < wnd_side; y_++){
				for(int x_ = 0; x_ < wnd_side; x_++){
					int x_start = x_local + x_;
					int y_start = y_local + y_;
					cost_aggr += cache[y_start * length + x_start]; 
				}
			}
			cost_aggr = (float)(cost_aggr)/(float)(wndwSize);
			ref_cost[base + cdim * y + x] = cost_aggr;
			int temp_x = 0;
			if(x-d<0) temp_x = x - d + cdim;
			else temp_x = x-d;
			tar_cost[base + (temp_x+cdim*y)] = cost_aggr;			
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
					costAggr += raw_cost[d * cdim * rdim + (xtar+cdim*ytar)];  
				}
			}
			costAggr = (float)(costAggr)/(float)(wndwSize);
			ref_cost[d * cdim * rdim + (x + cdim * y)] = costAggr;	
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
