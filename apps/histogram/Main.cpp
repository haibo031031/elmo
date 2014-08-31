#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "CLHelper.h"
#include "util.h"
#include <string.h>

#ifndef TIME
#define TIME
#endif

#ifndef VARIFY
#define VARIFY
#endif


void CPURun(unsigned int * data, unsigned int * hostBin, unsigned int width, unsigned int height);
void histogram_1(cl_mem d_data, cl_mem d_intermRes, unsigned int size, unsigned int bins, unsigned int wg_size);
void histogram_2(cl_mem d_data, cl_mem d_intermRes, unsigned int size, unsigned int bins, unsigned int wg_size);
void histogram_3(cl_mem d_data, cl_mem d_intermRes, unsigned int size, unsigned int bins, unsigned int wg_size);
void histogram_4(cl_mem d_data, cl_mem d_intermRes, unsigned int size, unsigned int bins, unsigned int wg_size);
void calcHistTot(unsigned int * deviceBin, unsigned int * subDeviceBin, unsigned int bins, unsigned int subHistCnt);

int main(int argc, char ** argv)
{
	unsigned int * data = NULL, * hostBin = NULL, * intermRes = NULL, * deviceBin = NULL;
	cl_mem d_data = NULL, d_intermRes = NULL;
try{
	if(argc!=4){
		printf("need 3 parameter here!!!");
		exit(-1);
	}

	_clInit(1, "gpu", 0);
	unsigned int iter = 100;
	
#if defined TIME
	double start_time = 0;
	double end_time = 0;
	string dat_name="data.dat";

	FILE * fp = fopen(dat_name.c_str(), "a+");
	if(fp==NULL)
	{
		printf("failed to open file!!!\n");
		exit(-1);
	}
#endif
	
	// parameters
	unsigned int side = atoi(argv[1]);
	unsigned int height = side;
	unsigned int width = side;
	unsigned int size = height * width;
	unsigned int wg_size = atoi(argv[2]);
	unsigned int bins = atoi(argv[3]);
	printf("size=%d, work-group size=%d, bins=%d\n", size, wg_size, bins);
	unsigned int nIntermRes = width * height / wg_size * bins;
	
	// allocate memory space on the host and device side
	data = (unsigned int * )malloc(width * height * sizeof(unsigned int));
	hostBin = (unsigned int * )malloc(bins * sizeof(unsigned int));
	deviceBin = (unsigned int * )malloc(bins * sizeof(unsigned int));
	intermRes = (unsigned int * )malloc(nIntermRes * sizeof(unsigned int));
	
	d_data = _clMalloc(width * height * sizeof(unsigned int));
	d_intermRes = _clMalloc( nIntermRes * sizeof(unsigned int));
	
	memset(hostBin, 0, bins * sizeof(unsigned int));
	memset(deviceBin, 0, bins * sizeof(unsigned int));
	//memset(intermRes, 0, nIntermRes * sizeof(unsigned int));
	fill<unsigned int>(data, size, bins);
	
	// copy data from host to device
	_clMemcpyH2D(d_data, data, size * sizeof(unsigned int));
	//memset(data, 0, size * sizeof(unsigned int));
	
#ifdef VARIFY	
	CPURun(data, hostBin, width, height);	
#endif //VARIFY
	
	/**************************1****************************/
	_clMemcpyH2D(d_intermRes, intermRes, nIntermRes * sizeof(unsigned int));
#ifdef TIME
	double deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		histogram_1(d_data, d_intermRes, size, bins, wg_size);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(intermRes, d_intermRes, nIntermRes * sizeof(unsigned int));
	calcHistTot(deviceBin, intermRes, bins, size / wg_size);
	verify_array_int<unsigned int>(deviceBin, hostBin, bins);
#endif //VARIFY

	memset(deviceBin, 0, bins * sizeof(unsigned int));


	/**************************2****************************/
	_clMemcpyH2D(d_intermRes, intermRes, nIntermRes * sizeof(unsigned int));
#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		histogram_2(d_data, d_intermRes, size, bins, wg_size);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(intermRes, d_intermRes, nIntermRes * sizeof(unsigned int));
	calcHistTot(deviceBin, intermRes, bins, size / wg_size);
	verify_array_int<unsigned int>(deviceBin, hostBin, bins);
#endif //VARIFY 

	memset(deviceBin, 0, bins * sizeof(unsigned int));
	/**************************3****************************/
	_clMemcpyH2D(d_intermRes, intermRes, nIntermRes * sizeof(unsigned int));
#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		histogram_3(d_data, d_intermRes, size, bins, wg_size);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(intermRes, d_intermRes, nIntermRes * sizeof(unsigned int));	
	calcHistTot(deviceBin, intermRes, bins, size / wg_size);
	verify_array_int<unsigned int>(deviceBin, hostBin, bins);
#endif //VARIFY

	memset(deviceBin, 0, bins * sizeof(unsigned int));

	memset(deviceBin, 0, bins * sizeof(unsigned int));
	/**************************4****************************/
	_clMemcpyH2D(d_intermRes, intermRes, nIntermRes * sizeof(unsigned int));
#ifdef TIME
	deltaT = 0.0;
#endif
	for(int i=0; i<iter; i++)
	{
	
#ifdef TIME
	start_time = gettime();
#endif

		histogram_4(d_data, d_intermRes, size, bins, wg_size);
	
#ifdef TIME	
	end_time = gettime();
	deltaT += end_time - start_time;	
#endif
	}	
#ifdef TIME
	fprintf(fp, "%lf\t", deltaT/(double)iter);
#endif

#ifdef VARIFY
	_clMemcpyD2H(intermRes, d_intermRes, nIntermRes * sizeof(unsigned int));
	calcHistTot(deviceBin, intermRes, bins, size / wg_size);
	verify_array_int<unsigned int>(deviceBin, hostBin, bins);
#endif //VARIFY 

#ifdef TIME	
	fprintf(fp, "\n");	
	fclose(fp);
#endif	
}
catch(string msg){
	printf("ERR:%s\n", msg.c_str());
	printf("Error catched\n");
	}

	_clFree(d_data);
	_clFree(d_intermRes);
	_clRelease();
	if(data!=NULL) free(data);
	if(hostBin!=NULL) free(hostBin);
	if(intermRes!=NULL) free(intermRes);
	if(deviceBin!=NULL) free(deviceBin);

	return 1;
}

void calcHistTot(unsigned int * deviceBin, unsigned int * midDeviceBin, unsigned int bins, unsigned int subHistCnt)
{
	for(unsigned int i = 0; i < subHistCnt; ++i)
    {
        for(unsigned int j = 0; j < bins; ++j)
        {
            deviceBin[j] += midDeviceBin[i * bins + j];
        }
    }
	return ;
}

void CPURun(unsigned int * data, unsigned int * hostBin, unsigned int width, unsigned int height)
{
    for(unsigned int i = 0; i < height; ++i)
    {
        for(unsigned int j = 0; j < width; ++j)
        {
            hostBin[data[i * width + j]]++;
        }
    }
}

void histogram_1(cl_mem d_data, cl_mem d_intermRes, unsigned int size, unsigned int bins, unsigned int wg_size){

	unsigned int range_x = size/bins;
	unsigned int range_y = 1;
	unsigned int group_x = wg_size;
	unsigned int group_y = 1;

	unsigned int kernel_id = 0;
	unsigned int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_data);
	_clSetArgs(kernel_id, arg_idx++, NULL, bins * wg_size * sizeof(unsigned char));
	_clSetArgs(kernel_id, arg_idx++, d_intermRes);
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(unsigned int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void histogram_2(cl_mem d_data, cl_mem d_intermRes, unsigned int size, unsigned int bins, unsigned int wg_size){

	unsigned int range_x = size/bins;
	unsigned int range_y = 1;
	unsigned int group_x = wg_size;
	unsigned int group_y = 1;

	unsigned int kernel_id = 1;
	unsigned int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_data);
	_clSetArgs(kernel_id, arg_idx++, NULL, bins * wg_size * sizeof(unsigned char));
	_clSetArgs(kernel_id, arg_idx++, d_intermRes);
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(unsigned int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void histogram_3(cl_mem d_data, cl_mem d_intermRes, unsigned int size, unsigned int bins, unsigned int wg_size){

	unsigned int range_x = size/bins;
	unsigned int range_y = 1;
	unsigned int group_x = wg_size;
	unsigned int group_y = 1;

	unsigned int kernel_id = 2;
	unsigned int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_data);
	_clSetArgs(kernel_id, arg_idx++, NULL, bins * wg_size * sizeof(unsigned char));
	_clSetArgs(kernel_id, arg_idx++, d_intermRes);
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(unsigned int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}

void histogram_4(cl_mem d_data, cl_mem d_intermRes, unsigned int size, unsigned int bins, unsigned int wg_size){

	unsigned int range_x = size/bins;
	unsigned int range_y = 1;
	unsigned int group_x = wg_size;
	unsigned int group_y = 1;

	unsigned int kernel_id = 3;
	unsigned int arg_idx = 0;
	_clSetArgs(kernel_id, arg_idx++, d_data);
	_clSetArgs(kernel_id, arg_idx++, NULL, bins * wg_size * sizeof(unsigned char));
	_clSetArgs(kernel_id, arg_idx++, d_intermRes);
	_clSetArgs(kernel_id, arg_idx++, &bins, sizeof(unsigned int));
	
	_clInvokeKernel2D(kernel_id, range_x, range_y, group_x, group_y);
	
	return ;
}
