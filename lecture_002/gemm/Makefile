
NVCC        = /usr/local/cuda/bin/nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = gemm
OBJ	        = gemm.o

default: $(EXE)

gemm.o: gemm.cu
	$(NVCC) -c -o $@ gemm.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
