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
	int pWidth; //width of parent matrix 
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
	//row of A determines C row, Col of B determines C col
	
	//Block determines which submatrix of C we work on
	//Create sub matrix of C to calculate with shared memory
	struct Matrix C_sub;
	C_sub.width = BLOCK_DIM; 
	C_sub.height = BLOCK_DIM;

	//int C_stride = C.width;
	int C_y = C.width * BLOCK_SIZE * blockIdx.y;
	int C_x = BLOCK_SIZE * blockIdx.x;
	C_sub.data = &C.data[C_y + C_x]

	//Thread determines where in C block we are
	float C_val = 0.0;
	int x = threadIdx.y;
	int y = threadIdx.x;	

	//loop over A and B submatrices to compute C submatrix
	for(int s = 0; s < (A.width / BLOCK_SIZE); m++){
		struct Matrix A_sub;
		A_sub.width = BLOCK_SIZE;
		A_sub.height = BLOCK_SIZE;
		int A_y = A.width * blockIdx.y * BLOCK_SIZE;
		int A_x = m * BLOCK_SIZE;
		A_sub.data = &A.data[A_y + A_x];
		
			
		struct Matrix B_sub;
		B_sub.width = BLOCK_SIZE;
		B_sub.height = BLOCK_SIZE;
		int B_y = B.width * m * BLOCK_SIZE;
		int B_x = blockIdx.x * BLOCK_SIZE;
		B_sub.data = &A.data[B_y + B_x];
		//this memory is shared between threads
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
	
		//each thread loads an element
		//note we use parent widths 	
		As[y][x] = A_sub.data[A.width * y + x];
		Bs[y][x] = B_sub.data[B.width * y + x];

		//make sure all memory is loaded
		__syncthread();

		//Compute Asub and Bsub product to accumulate Csub element
		for(int c = 0; c < BLOCK_SIZE; c++){
			C_val += As[y][c] * Bs[c][col];
		}	

		//wait for computation to finish before loading new memory
		__syncthread();	
	}	
	
	//write C sub element, again note parent width
	C_sub.data[C.width * y + col] = C_val;
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
		A.height = i*100;
		A.width = i*75;
		A.data = (float*)malloc(A.width * A.height * sizeof(float));
		InitMatrix(A);
		B.height = i*75;
		B.width = i*125;
		B.data = (float*)malloc(B.width * B.height * sizeof(float));
		InitMatrix(B);

		//Get Matrix Product of Array
		printf("********Entering Matrix Mul*****\n");
		struct Matrix C = MatMul(A, B);	
		free(A.data);
		free(B.data);
		free(C.data);
	}		
	
		
}
