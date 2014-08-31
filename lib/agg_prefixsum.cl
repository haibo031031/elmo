/** 
 * @brief   perform the scan operation on the local memory 
 * @param   gm_in: input data elements
 * @param   gm_out: output data elements
 * @param   lm: used local memory space
 * @param   num: the number of elements per threads (num=2 in the demo)
 * @param   ops: (0) sum; (1) max; (2) min; (4) others 
 */
void _clLMPrefixSum(__global float * input, __global float * output, __local float * block, uint num, uint ops)
//void _clLMScan(__global float *gm_in, __global float * gm_out, __local float * lm, uint num, uint ops)
{
	uint tid = get_local_id(0);
	uint tile = get_local_size(0);
	
	uint offset = 1;
	uint length = tile * num;
	
    /* Cache the computational window in shared memory */
	block[2*tid]     = input[2*tid];
	block[2*tid + 1] = input[2*tid + 1];	

    /* build the sum in place up the tree */
	for(uint d = length>>1; d > 0; d >>=1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(tid<d)
		{
			uint ai = offset*(2*tid + 1) - 1;
			uint bi = offset*(2*tid + 2) - 1;
			
			block[bi] += block[ai];
		}
		offset *= 2;
	}

    /* scan back down the tree */

    /* clear the last element */
	if(tid == 0)
	{
		block[length - 1] = 0;
	}

    /* traverse down the tree building the scan in the place */
	for(uint d = 1; d < length ; d *= 2)
	{
		offset >>=1;
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(tid < d)
		{
			uint ai = offset*(2*tid + 1) - 1;
			uint bi = offset*(2*tid + 2) - 1;
			
			float t = block[ai];
			block[ai] = block[bi];
			block[bi] += t;
		}
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

    /*write the results back to global memory */
	output[2*tid]     = block[2*tid];
	output[2*tid + 1] = block[2*tid + 1];	
}

