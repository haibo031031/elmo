//////////////////////////////////////////////////////////////////////
//Name: PixelBasedCost.cl 
//Created date: 15-11-2011
//Modified date: 29-1-2012
//Author: Gorkem Saygili, Jianbin Fang and Jie Shen
//Discription: calculates pixel based cost of stereo image - opencl kernel
/////////////////////////////////////////////////////////////////////

__kernel void pixelBasedCostL2R(const __global unsigned char * ref_image, const __global unsigned char * tar_image, __global float * cost, 
	const int dispRange, const int rdim, const int cdim){
	int x = get_global_id(0);//Thread Index: x dimension
	int y = get_global_id(1);//Thread Index: y dimension
	if(x<cdim && y<rdim){
		int idx = y * cdim + x;
		for(int d=0; d<dispRange; d++) {
			int tarc = x-d;
			if(tarc<0) tarc += cdim;
			//cost[d+dispRange*idx] = abs(ref_image[idx]-tar_image[y * cdim + tarc]);
			cost[d * cdim * rdim + idx] = abs(ref_image[idx]-tar_image[y * cdim + tarc]);
		}
	}
}


__kernel void pixelBasedCostR2L(const __global unsigned char *ref_image, 
	const __global unsigned char *tar_image, __global float *cost, 
	const int dispRange, const int rdim, const int cdim){
	
	int x = get_global_id(0);
	int y = get_global_id(1);
	if(x<cdim && y<rdim){
		int idx = y*cdim+x, xtar = 0;
		for(int d=0; d<dispRange; d++){
			int xtar = x+d;
			if(xtar>=cdim) xtar-=cdim;
			cost[d+dispRange*idx] = abs(ref_image[idx]-tar_image[y*cdim + xtar]);
		}
	}
}
/*
	revised by jianbin on 02/03 on
	+ storage format
*/
__kernel void pixelBasedCostL2R_Float(const __global float *ref_image, 
	const __global float *tar_image, __global float *cost, 
	const int dispRange, const int rdim, const int cdim){
	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int idx = y*cdim+x, xtar = 0;
	float res = 0;
	if(x<cdim && y<rdim){
		for(int d=0; d<dispRange; d++){
			xtar = x-d;
			if(xtar<0) xtar+=cdim;
			res = ref_image[idx]-tar_image[y * cdim + xtar];
			if(res<0) res=-res;
			//cost[d+dispRange*idx] = res;
			cost[d * cdim * rdim + idx] = res;
		}
	}
}


__kernel void pixelBasedCostR2L_Float(const __global float *ref_image, 
	const __global float *tar_image, __global float *cost, 
	const int dispRange, const int rdim, const int cdim){

	int x = get_global_id(0);
	int y = get_global_id(1);
	int idx = y*cdim+x,xtar = 0;
	float res = 0;
	for(int d=0; d<dispRange; d++){
		xtar = x+d;
		if(xtar>=cdim) xtar-=cdim;
		res = ref_image[idx]-tar_image[y * cdim + xtar];
		if(res<0) res=-res;
			cost[d+dispRange*idx] = res;
	}
}

/*
	revised by Jianbin on 27/02/2012
	on matrix storage format
*/
__kernel void pixelBasedCostL2R_Color(const __global unsigned char *ref_image, 
	const __global unsigned char *tar_image, __global float *cost, 
	const int dispRange, const int rdim, const int cdim,  const int normOrNot){
	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int idx = y*cdim+x, idxtar = 0, xtar = 0;
	if(x<cdim && y<rdim){
		for(int d=0; d<dispRange; d++){
			xtar = x-d;
			if(xtar<0) xtar+=cdim;
			idxtar = y*cdim + xtar;
			float coeff = (float)1/(float)3;
			float temp_cost = 0.0;
			if(normOrNot == 0)
				//cost[d+dispRange*idx] = abs(ref_image[idx*3]-tar_image[idxtar*3]) + \
										abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+ \
										abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]);
				temp_cost = abs(ref_image[idx*3]-tar_image[idxtar*3]) + \
										abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+ \
										abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]);
	
			else
				//cost[d+dispRange*idx] = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) + \
										abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+ \
										abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
				 temp_cost = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) + \
										abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+ \
										abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
			cost[d * rdim * cdim + idx] = temp_cost;
		}
	}
}

/*
	revised by Jianbin on 27/02/2012
	on matrix storage format
*/

__kernel void pixelBasedCostL2R_Color_Trunc(const __global unsigned char *ref_image, 
	const __global unsigned char *tar_image, __global float *cost, 
	const int dispRange, const int rdim, const int cdim, const float trunc, const int normOrNot){

	int x = get_global_id(0);
	int y = get_global_id(1);
	int idx = y*cdim+x, idxtar = 0, xtar = 0;
	float tempCost = 0;
	if(x<cdim && y<rdim){
		for(int d=0; d<dispRange; d++){
			xtar = x-d;
			if(xtar<0) xtar+=cdim;
			idxtar = y*cdim + xtar;
			float coeff = (float)1/(float)3;
			if(normOrNot == 0)
				tempCost = abs(ref_image[idx*3]-tar_image[idxtar*3]) +
						   abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) +
						   abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]);
			else
				tempCost = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
						          abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
								  abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
			if(trunc<tempCost)
				//cost[d+dispRange*idx] = trunc;
				//cost[d * rdim * cdim + idx] = trunc;
				tempCost = trunc;
			//else
				//cost[d+dispRange*idx] = tempCost;
				//cost[d * rdim * cdim + idx] = tempCost;
			cost[d * rdim * cdim + idx] = tempCost;
		}
	}
}

__kernel void pixelBasedCostR2L_Color(const __global unsigned char *ref_image, 
	const __global unsigned char *tar_image, __global float *cost, 
	const int dispRange, const int rdim, const int cdim, const int normOrNot){
	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int idx = y*cdim+x, idxtar = 0, xtar = 0;
	for(int d=0; d<dispRange; d++){
		xtar = x+d;
		if(xtar>=cdim) xtar-=cdim;
		idxtar = y*cdim + xtar;
		float coeff = (float)1/(float)3;
		if(normOrNot==0)
			cost[d+dispRange*idx] = (abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) + 
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
		else
			cost[d+dispRange*idx] = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
									abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) + 
									abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
	}
}

__kernel void pixelBasedCostR2L_Color_Trunc(const __global unsigned char *ref_image, 
	const __global unsigned char *tar_image, __global float *cost, 
	const int dispRange, const int rdim, const int cdim, const float trunc, const int normOrNot){

	int x = get_global_id(0);
	int y = get_global_id(1);
	int idx = y*cdim+x, idxtar = 0, xtar = 0;
	float tempCost = 0;
	idx = y*cdim+x;
	for(int d=0; d<dispRange; d++){
		xtar = x+d;
		if(xtar>=cdim) xtar-=cdim;
		idxtar = y*cdim + xtar;
		float coeff = (float)1/(float)3;
		if(normOrNot == 0)
			tempCost = abs(ref_image[idx*3]-tar_image[idxtar*3]) +
					   abs(ref_image[idx*3+1]-tar_image[idxtar*3+1]) +
					   abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]);
		else
			tempCost = coeff*(abs(ref_image[idx*3]-tar_image[idxtar*3]) +
					          abs(ref_image[idx*3+1]-tar_image[idxtar*3+1])+
							  abs(ref_image[idx*3+2]-tar_image[idxtar*3+2]));
		if(trunc<tempCost)
			cost[d+dispRange*idx] = trunc;
		else
			cost[d+dispRange*idx] = tempCost;
	}
}
