
//From CUDA By Example
//Running in HPC
//Request a compute node: srun -p gpu --gres=gpu:1 --pty bash
// Load cuda module: module load cuda/10.0
//compile: nvcc -o hello hello.cu
//Run: ./hello
//#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

//function name can be replaced with something meanigful. 
//The qualifier __global__ alerts the compiler that a function
//should be compiled to run on a device instead of the host
//__global__ void kernel (void){
__global__ void hello_world (int *out, int n){
    for (int i = 0; i < n; i++) //each GPU thread is working on each iteration
     out[i] = i;               
    }

int main(void) {

int *h_out;
int *d_out;

//Allocate Host memory
  h_out = (int*)malloc(sizeof (int) * N);


// Allocate device memory
    cudaMalloc((void**)&d_out, sizeof(int) * N);

// The angle brackets denotes agruments that we plan to pass to the running system
// The first number in the angle bracket represents number of parallel blocks 
//in which we would like the device to execute our kernel
//  the second parameter actually represents the number of threads per block 
// So, # of parallel threads = # of blocks x # of thread/block = 1

//    kernel<<<1,1>>>();
    hello_world<<<1,N>>>(d_out,N);
    
 // Transfer data back to host memory
    cudaMemcpy(h_out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);   

    for (int j=0; j<N; j++) {
      printf("Hello World from GPU Thread %d out of %d threads\n",h_out[j],N);
    }
   
  // Deallocate device memory
    cudaFree(d_out);
    // Deallocate host memory
    free(h_out); 
    
    return 0;
}
