/*
 * 5KK73
 * Eindhoven University of Technology
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 http://www.es.ele.tue.nl/~mwijtvliet/5KK73/?page=mmcuda
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Thread block size
#define BLOCK_SIZE 16

// outer product vetor size is VECTOR_SIZE * BLOCK_SIZE
#define VECTOR_SIZE 4

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA (32 * BLOCK_SIZE) // Matrix A width i.e. number of columns
#define HA (16 * BLOCK_SIZE) // Matrix A height i.e. number of rows
#define WB (24 * BLOCK_SIZE) // Matrix B width
#define HB WA  // Matrix B height should be same as the width of matrix A
#define WC WB  // Output Matrix C's width will be equal to Matrix B's width 
#define HC HA  // Matrix C's height is equal to the height of Matrix A


////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
matrixMul_naive( float* C, float* A, float* B, int wA, int wB)
{
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Accumulate row i of A and column j of B
  int row = by * blockDim.y + ty;
  int col  = bx * blockDim.x + tx;

  float accu = 0.0;

  for(int k=0; k<wA; k++){
    accu = accu + A[ row * wA + k ] * B[ k * wB + col ];
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
  C[ row * wB + col ] = accu;

}

int main(){

dim3 threads,grid;


// allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

// allocate device memory
   float* d_A;
    cudaMalloc((void**) &d_A, mem_size_A);
    float* d_B;
    cudaMalloc((void**) &d_B, mem_size_B);



// allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cudaMalloc((void**) &d_C, mem_size_C);

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);


// setup execution parameters
    threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    grid = dim3(WC / threads.x, HC / threads.y);
    // copy host memory to device
    cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice);
    // naive implementation
    matrixMul_naive<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);
    // copy result from device to host
    cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost);
// clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
