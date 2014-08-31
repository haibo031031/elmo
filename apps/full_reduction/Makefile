#C compiler
CC = g++
#0
#CUDA_DIR = /usr/local/cuda/
#INC = $(CUDA_DIR)/include/
#LIB = $(CUDA_DIR)/lib64/

#APP_DIR = /cm/shared/package/amd-app-sdk/2.6/
#INC = $(APP_DIR)/include/
#LIB = $(APP_DIR)/lib/x86_64
#2
#INTEL_DIR = /cm/shared/package/intel-ocl-sdk/1.1
#INC = $(INTEL_DIR)/include/
#LIB = $(INTEL_DIR)/lib64

#3
#NV_DIR = /cm/shared/apps/cuda41/toolkit/current/
#INC = $(NV_DIR)/include/
#LIB = $(NV_DIR)/lib/x86_64

SRC = Main.cpp
EXE = reduction

release: $(SRC)
	$(CC) -lOpenCL  -fopenmp $(SRC) -o $(EXE) -O3 -D ERRMSG

run:
	./$(EXE)
	
clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
