
#define GROUP_SIZE 16

__kernel void disp_refinement(__global unsigned char *dispIm, __global int *HorzOffset, __global int *VertOffset, 
	int rdim, int cdim, int dispRange){

	int idx,idx_left,idx_right,idx_top,idx_bottom,idx_ref;
	int tOffset,bOffset,lOffset,rOffset;
	int cury,curx_l,curx_r,dispVal;	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int x_local = get_local_id(0);
	int y_local = get_local_id(1);
	int dispHist[16]; //this is actually a bug!!!
	if(x<cdim && y<rdim){
		idx =y*cdim+x;
		tOffset = VertOffset[2*idx];
		bOffset = VertOffset[2*idx+1];
		for(int k=0; k<dispRange; k++)
			dispHist[k] = 0;
		for(int y_=-tOffset; y_<=bOffset; y_++){
			cury = y+y_;
			if((cury >= 0) && (cury < rdim)){
				idx_ref = x + cury*cdim;
				lOffset = HorzOffset[2*idx_ref];
				rOffset = HorzOffset[2*idx_ref+1];
				curx_l = x-lOffset;
				curx_r = x+rOffset;
				if(curx_l < 0)
					curx_l = 0;
				else if(curx_r >= cdim)
					curx_r = cdim-1;
				for(int x_=curx_l; x_<=curx_r; x_++){
					idx_ref = cury*cdim+x_;
					dispVal = dispIm[idx_ref];
					dispHist[dispVal]++;
				}
			}
		}

		int maxVal = dispHist[0];
		int maxDisp = 0;
		for(int i=1; i<dispRange; i++){
			if(dispHist[i]>maxVal){
				maxVal = dispHist[i];
				maxDisp = i;
			}
		}
		dispIm[idx] = (unsigned char)maxDisp;

	}	
}


__kernel void disp_refinement_lm(__global unsigned char *dispIm, __global int *HorzOffset, __global int *VertOffset, 
	int rdim, int cdim, int dispRange, __local int * dispHist){

	int idx,idx_left,idx_right,idx_top,idx_bottom,idx_ref;
	int tOffset,bOffset,lOffset,rOffset;
	int cury,curx_l,curx_r,dispVal;	
	int x = get_global_id(0);
	int y = get_global_id(1);
	int x_local = get_local_id(0);
	int y_local = get_local_id(1);
	//__local int dispHist[rdim][cdim][16]; //this is actually a bug!!!
	if(x<cdim && y<rdim){
		idx =y*cdim+x;
		int base = ((y_local * GROUP_SIZE) + x_local) * dispRange;
		tOffset = VertOffset[2*idx];
		bOffset = VertOffset[2*idx+1];
		for(int k=0; k<dispRange; k++)
			dispHist[base + k] = 0;
		for(int y_=-tOffset; y_<=bOffset; y_++){
			cury = y+y_;
			if((cury >= 0) && (cury < rdim)){
				idx_ref = x + cury*cdim;
				lOffset = HorzOffset[2*idx_ref];
				rOffset = HorzOffset[2*idx_ref+1];
				curx_l = x-lOffset;
				curx_r = x+rOffset;
				if(curx_l < 0)
					curx_l = 0;
				else if(curx_r >= cdim)
					curx_r = cdim-1;
				for(int x_=curx_l; x_<=curx_r; x_++){
					idx_ref = cury*cdim+x_;
					dispVal = dispIm[idx_ref];
					dispHist[base + dispVal]++;
				}
			}
		}

		int maxVal = dispHist[base + 0];
		int maxDisp = 0;
		for(int i=1; i<dispRange; i++){
			if(dispHist[base + i]>maxVal){
				maxVal = dispHist[base + i];
				maxDisp = i;
			}
		}
		dispIm[idx] = (unsigned char)maxDisp;

	}	
}


__kernel void borderExtrapolation(__global unsigned char *dispIm, __global unsigned char *crsschc, 
	int rdim, int cdim, int dispRange){

	int idx,idx_t;
	int flag = 0;
	int y = get_global_id(1);
	//for(int y=0; y<rdim; y++){
	if(y<rdim){
		flag = 0;
		for(int x=0; x<dispRange+20; x++){
			idx = y*cdim+x;
			if((dispIm[idx]>0) && (crsschc[idx]>0) && ((x-dispIm[idx]>0))){
				if(flag==0)
					for(int x_=0; x_<x; x_++){
						idx_t = y*cdim+x_;
						dispIm[idx_t] = dispIm[idx];
					}
				flag=1;
			}
		}
	}
}
