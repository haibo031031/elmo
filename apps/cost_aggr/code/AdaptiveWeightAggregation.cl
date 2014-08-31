//////////////////////////////////////////////////////////////////////
//Name: AdaptiveWeightAggregation.cl 
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Authors: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: aggregating cost with adaptive aggragation - opencl kernel
//////////////////////////////////////////////////////////////////////


//#include "Math.h"
#define square(x) x*x

float calcEuclideanDist3P(int l0, int l1, int a0, int a1, int b0, int b1){
	float diff0 = (float)l0-(float)l1;
	float diff1 = (float)a0-(float)a1;
	float diff2 = (float)b0-(float)b1;
	return sqrt(square(diff0) + square(diff1) + square(diff2));	
}

float calcEuclideanDist3P_float(float a1, float a2, float a3, float b1, float b2, float b3){
	float d1 = a1 - b1;
	float d2 = a2 - b2;
	float d3 = a3 - b3;
	return sqrt(d1*d1 + d2*d2 + d3*d3);
}


/*
	revised on 27/02/2012 by jianbin on
	+ storage format and coalesced access
	+ 
*/
__kernel void computeWeights(__global float *weights, 
	__global float *proxWeight, 
	const __global unsigned char *LABImage, 
	const float gamma_similarity, const int rdim, 
	const int cdim, const int maskRad){
	int i, j, l, m, ii, k, index, tar_index, pos_x, pos_y;
	float color_diff;
	float L1, a1, b1, L2, a2, b2;

	/* number of pixels in a image */
	int image_size=cdim*rdim;
	/* number of pixels in a window */
	int size=(2*maskRad+1)*(2*maskRad+1);

	int x = get_global_id(0);
	int y = get_global_id(1);
	// computation
	ii=y*cdim;
	if(x<cdim && y<rdim)//j=x_
	{
		index=ii+x;//y*cdim+x
		l=index*3;
		L1=(float)LABImage[l];
		a1=(float)LABImage[l+1];
		b1=(float)LABImage[l+2];
		for(k=0, i=-maskRad; i<=maskRad; i++)//y=i_
		{
			pos_y=y+i;
			//border check
			if(pos_y<0 || pos_y>=rdim)
				for(j=-maskRad; j<=maskRad; k++, j++)//x=j_
					weights[k+size*index]=0;
			else
			{
				pos_y*=cdim;
				for(j=-maskRad; j<=maskRad; k++, j++)
				{
					//border check
					//if(pSW[index][k]>0)
					//	continue;
					pos_x=x+j;
					if(pos_x<0 || pos_x>=cdim)
						weights[k+size*index]=0;
					else
					{
						tar_index=pos_x+pos_y;
						// color difference
						m=tar_index*3;
						L2=(float)LABImage[m];
						a2=(float)LABImage[m+1];
						b2=(float)LABImage[m+2];
						color_diff=calcEuclideanDist3P_float(L1, a1, b1, L2, a2, b2);
						//weights[k+size*index]=(float)(proxWeight[k]*exp(-color_diff/gamma_similarity));
						weights[k * cdim * rdim + index]=(float)(proxWeight[k]*exp(-color_diff/gamma_similarity));
						//pSW[tar_index][size-1-k]=pSW[index][k];
					}
				}
			}
		}
	}	
}

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
__kernel void calcAWCostL2R_lm(__global float *weightRef, 
	__global float *weightTar, __global float *raw_cost, 
	__global float *ref_cost, __global float *tar_cost, 
	const int rdim, const int cdim,	const int dispRange, const int maskRad, \
	__local float * cacher, const int tile_size, const int length){
	int y, x, k, l, d, n, index_ref, index1, index2, index_tar;
	float weight, weight_sum, sum;
	int maskArea = (2*maskRad+1)*(2*maskRad+1);
	index_ref=0;
	x = get_global_id(0);
	y = get_global_id(1);
	// aggregation 
	if(x<cdim && y <rdim)
	{
		for(d=0; d<dispRange; d++)
		{
			// load data from global memory to local memory
			_clWriteLM(raw_cost, cdim, rdim, maskRad, maskRad, cacher, length, length, d);
			int base = d * rdim * cdim;
			index_ref = cdim * y + x;			
			index_tar=x-d;
			//border check
			if(index_tar<0) index_tar=cdim+index_tar;
			//else if(index_tar>=cdim) index_tar=index_tar-cdim;
			index_tar+= cdim * y;
			weight_sum=0; 
			sum=0;
			n=0;
			for(k=-maskRad; k<=maskRad; k++)
			{
				index1=y+k;
				//border check
				if(index1<0) index1=rdim+index1;
				else if(index1>=rdim) index1=index1-rdim;
				index1*=cdim;
				for(l=-maskRad; l<=maskRad; l++)
				{
					index2=x+l;
					int base_2 = n * rdim * cdim;
					//border check
					if(index2<0) index2=cdim+index2;
					else if(index2>=cdim) index2=index2-cdim;
					weight_sum+=(weight=weightRef[base_2 + index_ref]*weightTar[base_2 + index_tar]);
					//sum+=raw_cost[base + (index1+index2)] * weight;
					//_clReadLM(__local float * cacher, int length_x, int offset_x, int offset_y, int cur_offset_x, int cur_offset_y)
					sum += _clReadLM(cacher, length, maskRad, maskRad, l, k) * weight;
					n++;
				}
			}
			ref_cost[base + index_ref]=(float)(sum/weight_sum);
			//tar_cost[base + index_tar]=(float)(sum/weight_sum);			
		}		
	}
}

/*
	revised on 27/02/2012 by jianbin on
	+ coalesced access
	+ using local memory
*/
/*__kernel void calcAWCostL2R_lm(__global float *weightRef, \
	__global float *weightTar, __global float *raw_cost, \
	__global float *ref_cost, __global float *tar_cost, \
	const int rdim, const int cdim,	const int dispRange, const int maskRad,\
	 __local float * cache, const int tile_size, const int length){
	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int x_local = get_local_id(0);
	int y_local = get_local_id(1);

	if(x<cdim && y <rdim)
	{
		int wnd_side = 2 * maskRad +1;
		int iter = (length%tile_size==0)?(length/tile_size):(length/tile_size+1);
		for(int d=0; d<dispRange; d++)
		{
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
			
			// + use cache data			
			int index_ref = cdim * y + x;
			int index_tar = x - d;		
			if(index_tar<0) index_tar = cdim + index_tar;
			index_tar += cdim * y;
			float weight_sum=0.0f, sum=0.0f, weight = 0.0f;
			int n=0;
			for(int kk=0; kk<wnd_side; kk++)
			{
				for(int ll=0; ll<wnd_side ; ll++)
				{
					int base_2 = n * rdim * cdim;
					weight = weightRef[base_2 + index_ref] * weightTar[base_2 + index_tar];
					weight_sum += weight;
					sum += cache[(y_local + kk ) * length + (x_local + ll)] * weight;
					n++;
				}
			}
			ref_cost[base + index_ref]=(float)(sum/weight_sum);
			tar_cost[base + index_tar]=(float)(sum/weight_sum);
		}
	}
}*/
__kernel void calcAWCostL2R(__global float *weightRef, 
	__global float *weightTar, __global float *raw_cost, 
	__global float *ref_cost, __global float *tar_cost, 
	const int rdim, const int cdim,	const int dispRange, const int maskRad){
	int y, x, k, l, d, n, index_ref, index1, index2, index_tar;
	float weight, weight_sum, sum;
	int maskArea = (2*maskRad+1)*(2*maskRad+1);
	index_ref=0;
	x = get_global_id(0);
	y = get_global_id(1);
	// aggregation 
	if(x<cdim && y <rdim)
	{
		for(d=0; d<dispRange; d++)
		{
			int base = d * rdim * cdim;
			index_ref = cdim * y + x;			
			index_tar=x-d;
			//border check
			if(index_tar<0) index_tar=cdim+index_tar;
			//else if(index_tar>=cdim) index_tar=index_tar-cdim;
			index_tar+= cdim * y;
			weight_sum=0; 
			sum=0;
			n=0;
			for(k=-maskRad; k<=maskRad; k++)
			{
				index1=y+k;
				//border check
				if(index1<0) index1=rdim+index1;
				else if(index1>=rdim) index1=index1-rdim;
				index1*=cdim;
				for(l=-maskRad; l<=maskRad; l++)
				{
					index2=x+l;
					int base_2 = n * rdim * cdim;
					//border check
					if(index2<0) index2=cdim+index2;
					else if(index2>=cdim) index2=index2-cdim;
					weight_sum+=(weight=weightRef[base_2 + index_ref]*weightTar[base_2 + index_tar]);
					sum+=raw_cost[base + (index1+index2)] * weight;
					n++;
				}
			}
			ref_cost[base + index_ref]=(float)(sum/weight_sum);
			//tar_cost[base + index_tar]=(float)(sum/weight_sum);			
		}		
	}
}


__kernel void calcAWCostR2L(__global float *weightRef, 
	__global float *weightTar, __global float *cost, 
	const int rdim, const int cdim,\
	const int dispRange, const int maskRad){

	int x = get_global_id(0);
	int y = get_global_id(1);
	int maskEdge = 2*maskRad+1;
	int maskArea = maskEdge*maskEdge;
	int idx=y*cdim+x, refidx=0, taridx=0, yref=0, xref=0, ytar=0,
		xtar=0,maskInd=0,refInd=0,tarInd=0;
	float weight,sum_weight,score;
	for(int d=0; d<dispRange; d++){
		sum_weight = 0;
		maskInd = 0;
		score = 0;
		for(int y_=-maskRad; y_<=maskRad; y_++){
			yref = y+y_;
			if(yref<0) yref+=rdim;
			else if(yref>=rdim) yref-=rdim;
			ytar = yref;
			for(int x_=-maskRad; x_<=maskRad; x_++){
				xref = x+x_;
				xtar = xref+d;
				if(xref<0) xref += cdim;
				else if(xref>=cdim) xref -= cdim;
				if(xtar<0) xtar += cdim;
				else if(xtar>=cdim) xtar-=cdim;
				refInd = yref*cdim+xref;
				tarInd = ytar*cdim+xtar;
				refidx = maskInd+maskArea*refInd;
				taridx = maskInd+maskArea*tarInd;
				weight = weightRef[refidx]*weightTar[taridx];
				//weight = 1;
				sum_weight += weight;
				//sum_weight = 1;
				score += cost[d+dispRange*refInd]*weight;
				//score += 1;
				maskInd++;
			}
		}
		if(sum_weight != 0)
			cost[d+dispRange*idx] = (float)(score/sum_weight);
		else
			cost[d+dispRange*idx] = (float)score;
	}
}
