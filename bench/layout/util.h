#ifndef _C_UTIL_
#define _C_UTIL_
#include <math.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>
/*
	timer (seconds)
*/
double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec * 1e+3 + t.tv_usec * 1e-3;
}
//-------------------------------------------------------------------
//--initialize array with maximum limit
//-------------------------------------------------------------------
template<typename datatype>
void fill(datatype *A, const int n, const datatype maxi){
    for (int j = 0; j < n; j++){
        A[j] = ((datatype) maxi * (rand() / (RAND_MAX + 1.0f)));
    }
}

//--print matrix
template<typename datatype>
void print_matrix(datatype *A, int height, int width){
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			int idx = i*width + j;
			std::cout<<A[idx]<<" ";
		}
		std::cout<<std::endl;
	}

	return;
}
//-------------------------------------------------------------------
//--verify results
//-------------------------------------------------------------------
#define MAX_RELATIVE_ERROR  .002
template<typename datatype>
void verify_array(const datatype *cpu_results, const datatype *gpu_results, const int w, const int h, const int b){

    bool passed = true; 
    int count = 0;
#pragma omp parallel for reduction(+ : count)
	for(int i=b; i<(h+b); i++){
		for(int j=b; j<(w+b); j++){
			if(fabs(cpu_results[i*(w+2*b)+j]-gpu_results[i*(w+2*b)+j])/cpu_results[i*(w+2*b)+j]>MAX_RELATIVE_ERROR){
	        	passed = false; 
		        std::cout<<cpu_results[i]<<"::"<<gpu_results[i]<<std::endl;
        		count++;
			}
		}
	}
    if (passed){
        std::cout << "--cambine:passed:-)" << std::endl;
    }
    else{
        std::cout << "--cambine: failed ("<<((double)count/(double)w*h)*100<<"%)..." << std::endl;
        
    }
    return ;
}

template<typename datatype>
void verify_array(const datatype *cpu_results, const datatype *gpu_results, const int size){

    bool passed = true; 
    int count = 0;
//#pragma omp parallel for reduction(+ : count)
    for (int i=0; i<size; i++){
      if (fabs(cpu_results[i] - gpu_results[i]) / cpu_results[i] > MAX_RELATIVE_ERROR){
         passed = false; 
         std::cout<<cpu_results[i]<<"::"<<gpu_results[i]<<std::endl;
         count++;
      }
    }
    if (passed){
        std::cout << "--cambine:passed:-)" << std::endl;
    }
    else{
        std::cout << "--cambine: failed ("<<((double)count/(double)size)*100<<"%)..." << std::endl;
        
    }
    return ;
}
template<typename datatype>
void verify_array_int(const datatype *cpu_results, const datatype *gpu_results, unsigned int w, unsigned int h){

    bool passed = true; 
    int count = 0;
//#pragma omp parallel for reduction(+ : count)
	for(unsigned int y=0; y<h; y++)
	{
		for(unsigned int x=0; x<w; x++)
		{
			unsigned int idx = y * w + x;
			if(cpu_results[idx] != gpu_results[idx])
			{
		         passed = false; 
		         std::cout<<"("<<x<<","<<y<<")"<<cpu_results[idx]<<"::"<<gpu_results[idx]<<std::endl;
		         count++;				
			}
		}
	}

    if (passed){
        std::cout << "--cambine:passed:-)" << std::endl;
    }
    else{
        std::cout << "--cambine: failed ("<<count<<")" << std::endl;
        
    }
    return ;
}


#endif

