//https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial02/
//Running in HPC
//Request a compute node: srun -p gpu --gres=gpu:1 --pty bash
// Load cuda module: module load cuda/10.0
//compile: nvcc -o serial hello.cu
//Run: ./serial
//Profile: nvprof ./serial

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000000
#define MAX_ERR 1e-6

//Each thread simply sums a single element from each of the two input vectors and writes the result into the output vector

__global__ void vector_add(float *out, float *a, float *b, int n) {
        
// Determine which element this thread is computing
	//int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	//int thread_id = blockDim.x * block_id + threadIdx.x;
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

// Compute a single element of the result vector (if the element is valid)
	if (thread_id < N) out[thread_id] = a[thread_id] + b[thread_id];
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);

    // Transfer data from host to device memory
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    
// Determine the number of thread blocks in the x- and y-dimension
	int threads_per_block = 1024; //from deviceQuery - (Maximum number of threads per block)
        int max_blocks_per_dimension = 65535; //from deviceQuery - Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  
        int num_blocks = (int) ((float) (N + threads_per_block - 1) / (float) threads_per_block); //blocks required for N independent vector addition operations
	int num_blocks_y = (int) ((float) (num_blocks + max_blocks_per_dimension - 1) / (float) max_blocks_per_dimension);
	int num_blocks_x = (int) ((float) (num_blocks + num_blocks_y - 1) / (float) num_blocks_y);
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);
	
// Execute the kernel to compute the vector sum on the GPU
	vector_add <<< grid_size , threads_per_block >>> (d_out, d_a, d_b, N);

    
    // Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

   // Display the Results
   // for (int i=0; i<N; i++) {
   //   printf("%d + %d = %d\n", a[i],b[i],out[i]);
   //}

    printf("PASSED\n");

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}

