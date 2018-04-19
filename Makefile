all:
	nvcc -g -G -o matMulGlobal matMulGlobal.cu
	nvcc -g -G -o matMulLocal matMulLocal.cu
