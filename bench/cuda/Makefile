#C compiler
CC = nvcc
#CUDA_DIR = /usr/local/cuda/
#INC = $(CUDA_DIR)/include/
#IB = $(CUDA_DIR)/lib/

SRC = copy.cu
EXE = copy

release: $(SRC)
	$(CC)  $(SRC) -o $(EXE)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result*
