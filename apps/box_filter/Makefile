#C compiler
CC = g++
#0
CUDA_DIR = /usr/local/cuda/
INC = $(CUDA_DIR)/include/
LIB = $(CUDA_DIR)/lib64/

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

SRC = Main.cpp shrUtils.cpp cmd_arg_reader.cpp oclUtils.cpp
EXE = app

release: $(SRC)
	$(CC) -lOpenCL  -fopenmp $(SRC) -o $(EXE) -I$(INC) -L$(LIB)

errmsg: $(SRC)
	$(CC) -lOpenCL -fopenmp $(SRC) -o $(EXE) -I$(INC) -L$(LIB) -D ERRMSG 

device: $(SRC)
	$(CC) -lOpenCL -fopenmp $(SRC) -o $(EXE) -I$(INC) -L$(LIB) -D ERRMSG -D DEV_INFO

ptx: $(SRC)
	$(CC) -lOpenCL -fopenmp $(SRC) -o $(EXE) -I$(INC) -L$(LIB) -D PTX_MSG

debug: $(SRC)
	$(CC) -g -lOpenCL $(SRC) -o $(EXE) -I$(INC) -L$(LIB)
	
run:
	./$(EXE)
	
clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
