#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>

//use 16 for portability
#define BLOCK_SIZE 16


struct Matrix
{
	int height;
	int width;
	float* data;
};

void InitMatrix(Matrix M);

Matrix CopyShape(const Matrix M){
	struct Matrix M_copy;
	M_copy.height = M.height;
	M_copy.width = M.width;
	return M_copy;
}

//A and B are input, C is output
//Naive matrix multiplication algorithm
__global__ void MatMul_k(const struct Matrix A, const struct Matrix B, Matrix C){
	//Block is 2d
	int A_row = blockDim.y * blockIdx.y + threadIdx.y;
	int B_col = blockDim.x * blockIdx.x + threadIdx.x;

	if(A_row > A.height || B_col > B.width)
		return;

	float entry = 0.0;
	for(int i=0; i < A.width; i++){
		entry += A.data[A_row + i] * B.data[B_col + i*B.width];
	}

	C.data[A_row*C.width + B_col] = entry;	
}

Matrix MatMul(const Matrix A, const Matrix B){
	Matrix C;
	C.width = A.height;
	C.height = B.width;	
	C.data = (float*)malloc(C.width * C.height * sizeof(float));

	if(A.width != B.height){
		printf("Inner matrix dimensions must be equal!");
		C.data = NULL;
		return C;
	}

	//Copy A and B over to GPU
	struct Matrix A_gpu;// = CopyShape(A);
	A_gpu.height = A.height;
	A_gpu.width = A.width;
	size_t A_size = A_gpu.height * A_gpu.width * sizeof(float);
	cudaError_t err = cudaMalloc(&A_gpu.data, A_size);
	printf("Cuda Error: malloc A: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(A_gpu.data, A.data, A_size, cudaMemcpyHostToDevice);
	printf("Cuda Error: cpy A: %s\n", cudaGetErrorString(err));

	struct Matrix B_gpu = CopyShape(B);
	size_t B_size = B_gpu.height * B_gpu.width * sizeof(float);
	err = cudaMalloc(&B_gpu.data, B_size);
	printf("Cuda Error: malloc B: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(B_gpu.data, B.data, B_size, cudaMemcpyHostToDevice);
	printf("Cuda Error: cpy B: %s\n", cudaGetErrorString(err));

	//Make space for resul matrix
	struct Matrix C_gpu = CopyShape(C);
	size_t C_size = C_gpu.width * C_gpu.height * sizeof(float); 	
	//C_gpu.data = (float*)malloc(C_gpu.width * C_gpu.height * sizeof(float));
	//InitMatrix(C_gpu);
	cudaMalloc(&C_gpu.data, C_size);
	printf("Cuda Error: malloc C: %s\n", cudaGetErrorString(err));
	err = cudaMemcpy(C_gpu.data, C.data, C_size, cudaMemcpyHostToDevice);
	printf("Cuda Error: cpy C: %s\n", cudaGetErrorString(err));

	//Run Cuda Code
	dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE); //z dim = 1
	int grid_x = ceil(C_gpu.width/block_dim.x);
	int grid_y = ceil(C_gpu.height/block_dim.y);
	dim3 grid_dim(grid_x, grid_y);
	MatMul_k<<<grid_dim, block_dim>>>(A_gpu, B_gpu, C_gpu);
	err = cudaThreadSynchronize();
	printf("Run Cuda Code: %s\n", cudaGetErrorString(err));

	//Get Result
	err = cudaMemcpy(C.data, C_gpu.data, C_size, cudaMemcpyDeviceToHost);
	printf("Get Result: %s\n", cudaGetErrorString(err));

	cudaFree(A_gpu.data);
	cudaFree(B_gpu.data);
	cudaFree(C_gpu.data);

	return C;
}

void SetVal(Matrix M, int x, int y, float val){
	if(y*M.width + x > M.width*M.height)
		printf("Reading past end of array\n");
	M.data[y*M.width + x] = val;
}

float GetVal(Matrix M, int x, int y){
	return M.data[y*M.width + x];
}

void InitMatrix(Matrix M){
	for(int y = 0; y<M.height; y++){
		for(int x = 0; x<M.width; x++){
			float val = 20*(float)rand()/(float)RAND_MAX;
			SetVal(M, x, y, val);
		}
	}	
}

int main(){
	int NUM_ARRAYS = 3;
	//struct Matrix As[NUM_ARRAYS];
	//struct Matrix Bs[NUM_ARRAYS];

	for(int i=1; i<NUM_ARRAYS+1; i++){
		struct Matrix A, B;
		//Initialize Array
		A.height = i*5000;
		A.width = i*3500;
		A.data = (float*)malloc(A.width * A.height * sizeof(float));
		InitMatrix(A);
		B.height = i*3500;
		B.width = i*7500;
		B.data = (float*)malloc(B.width * B.height * sizeof(float));
		InitMatrix(B);

		//Get Matrix Product of Array
		printf("********Entering Matrix Mul*****\n");
		clock_t start = clock();
		struct Matrix C = MatMul(A, B);	
		clock_t time = clock() - start;
		float sec = (float)time/(float)CLOCKS_PER_SEC;
		printf("Time %d: %f\n", i, sec);
		free(A.data);
		free(B.data);
		free(C.data);
	}		
	
		
}
