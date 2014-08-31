//////////////////////////////////////////////////////////////////////
//Name: CrossBasedAggregation.cl 
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: cross based aggregation- opencl kernel
/////////////////////////////////////////////////////////////////////

#define min(a,b) (a>b) ? b:a
#define max(a,b) (b>a) ? b:a 
#define L 17

void calc_x_range(const __global unsigned char * image, int y, int x, int rdim, int cdim, int * leftOffset, int * rightOffset, int tao){
	int idx_ref = y*cdim + x;
	int x_ = 0;
	unsigned char val_ref_g = image[idx_ref*3];
	unsigned char val_ref_b = image[idx_ref*3+1];
	unsigned char val_ref_r = image[idx_ref*3+2];
	unsigned char val_tar_g,val_tar_b,val_tar_r;
	float delta = 0, delta_;
	int idx_tar,tarx=0;
	while((tarx>=0) && (delta<tao) && (x_<L)){
		x_++;
		tarx = x-x_;
		if(tarx>=0){
			idx_tar = y*cdim + tarx;
			val_tar_g = image[idx_tar*3];
			val_tar_b = image[idx_tar*3+1];
			val_tar_r = image[idx_tar*3+2];
			delta_ = max(abs(val_ref_b - val_tar_b),abs(val_ref_r - val_tar_r));
			delta =  max(abs(val_ref_g - val_tar_g),delta_);
		}
		else x_--;
	}
	if(delta>tao)
		x_--;
	if((x_==0) && (x != 0))
		x_++;
	if((x_ <= 0) && (x != 0))
		x_ = 1;
	*leftOffset = x_;
	x_ = 0;
	delta = 0;
	tarx = 0;
	while((tarx<cdim) && (delta<tao) && (x_<L)){
		x_++;
		tarx = x+x_;
		if(tarx<cdim){
			idx_tar = y*cdim + tarx;
			val_tar_g = image[idx_tar*3];
			val_tar_b = image[idx_tar*3+1];
			val_tar_r = image[idx_tar*3+2];
			delta_ = max(abs(val_ref_b - val_tar_b),abs(val_ref_r - val_tar_r));
			delta =  max(abs(val_ref_g - val_tar_g),delta_);
		}
		else x_--;
	}
	if(delta>tao)
		x_--;
	if((x_ == 0) && (x < cdim-1))
		x_++;
	if((x_ <= 0) && (x != cdim-1))
		x_ = 1;
	*rightOffset = x_;
}

void calc_y_range(const __global unsigned char * image, int y, int x, int rdim, int cdim, int * topOffset, int * bottomOffset, int tao){
	int idx_ref = y*cdim + x;
	int y_ = 0;
	unsigned char val_ref_g = image[idx_ref*3];
	unsigned char val_ref_b = image[idx_ref*3+1];
	unsigned char val_ref_r = image[idx_ref*3+2];
	unsigned char val_tar_g,val_tar_b,val_tar_r;
	float delta = 0,delta_;
	int idx_tar,tary=0;
	while((tary>=0) && (delta<tao) && (y_<L)){
		y_++;
		tary = y-y_;
		if(tary>=0){
			idx_tar = tary*cdim + x;
			val_tar_g = image[idx_tar*3];
			val_tar_b = image[idx_tar*3+1];
			val_tar_r = image[idx_tar*3+2];
			delta_ = max(abs(val_ref_b - val_tar_b),abs(val_ref_r - val_tar_r));
			delta =  max(abs(val_ref_g - val_tar_g),delta_);
		}
		else y_--;
	}
	if(delta>tao)
		y_--;
	if((y_ == 0) && (y != 0))
		y_++;
	if((y_ <= 0) && (y != 0))
		y_=1;
	*topOffset = y_;
	y_=0;
	delta = 0;
	tary = 0;
	while((tary<rdim) && (delta<tao) && (y_<L)){
		y_++;
		tary = y+y_;
		if(tary<rdim){
			idx_tar = tary*cdim+x;
			val_tar_g = image[idx_tar*3];
			val_tar_b = image[idx_tar*3+1];
			val_tar_r = image[idx_tar*3+2];
			delta_ = max(abs(val_ref_b - val_tar_b),abs(val_ref_r - val_tar_r));
			delta =  max(abs(val_ref_g - val_tar_g),delta_);
		}
		else y_--;
	}
	if(delta>tao)
		y_--;
	if(y_ == 0 && y < rdim-1)
		y_++;
	if((y_ <= 0) && (y != rdim-1))
		y_=1;
	*bottomOffset = y_;
}

__kernel void findCross(const __global unsigned char *image, __global int *HorzOffset, __global int *VertOffset, int rdim, int cdim, int tao){
	int topOffset,bottomOffset,leftOffset,rightOffset,idx;
	float costVal = 0,min_cost;
	int x = get_global_id(0);//Thread Index: x dimension
	int y = get_global_id(1);//Thread Index: y dimension
	if(x<cdim && y<rdim){
		idx = y*cdim+x;
		calc_y_range(image, y, x, rdim, cdim, &topOffset, &bottomOffset, tao);
		VertOffset[2*idx] = topOffset;
		VertOffset[1+2*idx] = bottomOffset;
		calc_x_range(image, y, x, rdim, cdim, &leftOffset, &rightOffset, tao);
		HorzOffset[2*idx] = leftOffset;
		HorzOffset[1+2*idx] = rightOffset;
	}
}

/*
	revised by jianbin on 27/02/2012
	+ parallelism
	+ storage format and coalesced access
*/
__kernel void aggregate_cost_horizontal(__global float *cost_image, int rdim, int cdim, int dispRange){
	int d = get_global_id(0);
	int y = get_global_id(1);
	if(y<rdim){
		int base = d * rdim * cdim;
		for(int x=1; x<cdim; x++){
			int idx = y * cdim + x;
			//for(int d=0; d<dispRange; d++){
				//idx = y*cdim+x;
				//cost_image[d+dispRange*(y*cdim+x)] += cost_image[d+dispRange*(y*cdim+x-1)];
			cost_image[base + idx] += cost_image[ base + idx - 1];  
			//}	
		}
	}
}

/*
	revised by jianbin on 27/02/2012
	+ storage format and coalesced access
	+
*/
__kernel void cross_stereo_aggregation(const __global float * raw_cost, __global float *ref_cost, __global float * tar_cost,\
	__global int *HorzOffset_ref, __global int *HorzOffset_tar, __global int *VertOffset_ref, __global int *VertOffset_tar,  \
	int rdim, int cdim, int dispRange){
	int idx_ref,idx_tar,tarx,tOffset,bOffset,lOffset,rOffset,idx,idx_t,idx_left,idx_right;
	int cury, curx,curx_l,curx_r,cnt;
	float costSum;
	int x = get_global_id(0);
	int y = get_global_id(1);
	if(x<cdim && y<rdim){
		idx = x + y*cdim;
		for(int d=0; d<dispRange; d++){
			int base = d * rdim * cdim;
			tarx = x-d;
			if(tarx < 0) tarx += cdim;
			idx_t = tarx + y*cdim;
			tOffset = min(VertOffset_ref[2*idx],VertOffset_tar[2*idx_t]);
			bOffset = min(VertOffset_ref[1+2*idx],VertOffset_tar[1+2*idx_t]);
			costSum = 0;
			cnt = 0;
			for(int y_=-tOffset; y_<=bOffset; y_++){
				cury = y+y_;
				if(cury < 0) cury = 0;
				else if(cury >= rdim) cury = rdim - 1;

				idx_ref = x + cury*cdim;
				idx_tar = tarx + cury*cdim;
				lOffset = min(HorzOffset_ref[2*idx_ref],HorzOffset_tar[2*idx_tar]);
				rOffset = min(HorzOffset_ref[1+2*idx_ref],HorzOffset_tar[1+2*idx_tar]);
					
				curx_l = x-lOffset-1;
				curx_r = x+rOffset;
				if(curx_l < 0)
					curx_l = 0;
				else if(curx_r >= cdim)
					curx_r = cdim-1;

				idx_left = cury*cdim+curx_l;
				idx_right = cury*cdim+curx_r;
				costSum += (raw_cost[base + idx_right]-raw_cost[base + idx_left]);
				cnt += rOffset+lOffset+1;
			}
			costSum /= cnt;
			ref_cost[base + idx] = costSum;
			tar_cost[base + idx_t] = costSum;
		}
	}
}

/*
	- revised by jianbin on 15/03/2012 on
	+ using local memory
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
	return ;
}

/*
	subroutine: load data from global memory into local memory (TBT)
	extended by Jianbin Fang on May 11, 2012 with general borders:
	@params	bs_top
	@params	bs_bottom
	@params	bs_left
	@params	bs_right
	@params	pad_size
*/
void _clMemWriteBRO(__global float * gl_data, uint cdim, uint rdim, __local float * cacher, uint bs_top, uint bs_bottom, uint bs_left, uint bs_right, uint pad_size, uint d)
{
	uint t_gl_x = get_global_id(0);
	uint t_gl_y = get_global_id(1);
	uint t_lc_x = get_local_id(0);
	uint t_lc_y = get_local_id(1);
	uint wg_x = get_local_size(0);
	uint wg_y = get_local_size(1);
	
	//uint length_x = wg_x + 2 * radius;
	uint length_x = wg_x + bs_left + bs_right + pad_size;
	//uint length_y = wg_y + 2 * radius;
	uint length_y = wg_y + bs_top + bs_bottom;
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
					int x_tmp = (t_gl_x - bs_left) + i;
					if(x_tmp<0)	x_tmp = 0;
					else if(x_tmp>=cdim) x_tmp = cdim-1;
					int y_tmp = (t_gl_y - bs_top) + j;
					if(y_tmp<0)	y_tmp = 0;
					else if(y_tmp>=rdim) y_tmp = rdim-1;
					int id_global = d * cdim * rdim + y_tmp * cdim + x_tmp;
					cacher[id_local] = gl_data[id_global];
				}
			}
		}			
	}	
	barrier(CLK_LOCAL_MEM_FENCE);
	return ;
}

float _clReadLM(__local float * cacher, int length_x, int offset_x, int offset_y, int cur_offset_x, int cur_offset_y){
	int t_lc_x = get_local_id(0); 
	int t_lc_y = get_local_id(1); 
	int d_lc_x = t_lc_x + offset_x + cur_offset_x; 
	int d_lc_y = t_lc_y + offset_y + cur_offset_y; 
	return cacher[d_lc_y * length_x + d_lc_x]; 
}

#define OCL_PRAGMA_LM_R(cacher, length_x, offset_x, offset_y, cur_offset_x, cur_offset_y) _clReadLM(cacher, length_x, offset_x, offset_y, cur_offset_x, cur_offset_y)

__kernel void cross_stereo_aggregation_lm(const __global float * raw_cost, __global float *ref_cost, __global float * tar_cost,\
	__global int *HorzOffset_ref, __global int *HorzOffset_tar, __global int *VertOffset_ref, __global int *VertOffset_tar,  \
	int rdim, int cdim, int dispRange, __local float * cacher, const int tile_size, const int length_x, const int length_y, \
	const int offset_x, const int offset_y){
	int idx_ref,idx_tar,tarx,tOffset,bOffset,lOffset,rOffset,idx,idx_t,idx_left,idx_right;
	int cury, curx,curx_l,curx_r,cnt;
	float costSum;
	int x = get_global_id(0);
	int y = get_global_id(1);
	if(x<cdim && y<rdim){
		idx = x + y*cdim;
		for(int d=0; d<dispRange; d++){
			/*  loading data from global memory to local memory:
				#pragma ocl lm(raw_cost, cacher)
			*/
			//_clWriteLM(raw_cost, cdim, rdim, offset_x, offset_y, cacher, length_x, length_y, d);
			_clMemWriteBRO(raw_cost, cdim, rdim, cacher, L, L, (L+1), L, 0, d);
			int base = d * rdim * cdim;
			tarx = x-d;
			if(tarx < 0) tarx += cdim;
			idx_t = tarx + y*cdim;
			tOffset = min(VertOffset_ref[2*idx],VertOffset_tar[2*idx_t]);
			bOffset = min(VertOffset_ref[1+2*idx],VertOffset_tar[1+2*idx_t]);
			costSum = 0;
			cnt = 0;
			for(int y_=-tOffset; y_<=bOffset; y_++){
				cury = y+y_;
				if(cury < 0) cury = 0;
				else if(cury >= rdim) cury = rdim - 1;

				idx_ref = x + cury*cdim;
				idx_tar = tarx + cury*cdim;
				lOffset = min(HorzOffset_ref[2*idx_ref],HorzOffset_tar[2*idx_tar]);
				rOffset = min(HorzOffset_ref[1+2*idx_ref],HorzOffset_tar[1+2*idx_tar]);
					
/*				curx_l = x-lOffset-1;
				curx_r = x+rOffset;
				if(curx_l < 0)
					curx_l = 0;
				else if(curx_r >= cdim)
					curx_r = cdim-1;

				idx_left = cury*cdim+curx_l;
				idx_right = cury*cdim+curx_r;
*/
				//costSum += (raw_cost[base + idx_right]-raw_cost[base + idx_left]);
				//costSum += _clReadLM(cacher, length_x, offset_x, offset_y, rOffset, y_) - _clReadLM(cacher, length_x, offset_x, offset_y, -(lOffset+1), y_);
				costSum += OCL_PRAGMA_LM_R(cacher, length_x, offset_x, offset_y, rOffset, y_) - OCL_PRAGMA_LM_R(cacher, length_x, offset_x, offset_y, -(lOffset+1), y_);
				cnt += rOffset+lOffset+1;
			}
			barrier(CLK_LOCAL_MEM_FENCE);
			costSum /= cnt;
			ref_cost[base + idx] = costSum;
			tar_cost[base + idx_t] = costSum;
		}
	}
}
