// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>


#define N 160      // dimension of the Matrix a and Matrix b
#define MAX_THREADS_PER_BLOCK 1024 // From device query
#define THREADS 32 //1024 = 32x32
#
__global__ void matrixMult (int *a, int *b, int *c, int width) {
 int k, sum = 0;
 int col = threadIdx.x + blockDim.x * blockIdx.x;
 int row = threadIdx.y + blockDim.y * blockIdx.y;
 if(col < width && row < width) {
 for (k = 0; k < width; k++)
 sum += a[row * width + k] * b[k * width + col];
 c[row * width + col] = sum;
 }
}

int main() {
// int a[N][N], b[N][N], c[N][N];
 int *a,*b,*c;
 a = (int*)malloc(N*N*sizeof(int));
 b = (int*)malloc(N*N*sizeof(int));
 c = (int*)malloc(N*N*sizeof(int));
 int *dev_a, *dev_b, *dev_c;

 // initialize matrices a and b with appropriate values

// Initialize host arrays
    for( int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
        *(a + i*N + j)= 1;
        *(b + i*N + j)= 1;
        
    }
}
//Allocate appropriate memory for devices a, b, and c

 int size = N * N * sizeof(int);
 cudaMalloc((void **) &dev_a, size);
 cudaMalloc((void **) &dev_b, size);
 cudaMalloc((void **) &dev_c, size);
 cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

//select appropriate resources
 
int block_size = sqrt(MAX_THREADS_PER_BLOCK);
int grid_size = ceil((float)N/block_size); 

printf("block_size=%dx%d \t grid_size=%dx%d\n", block_size, block_size,grid_size,grid_size);

 dim3 dimGrid(grid_size,grid_size);
 dim3 dimBlock(block_size,block_size);

//Execute Kernel - Pass the Parameters
 matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
//Test
//matrixMult<<<1, 1024>>>(dev_a, dev_b, dev_c, N); 

//copy the data back to the device
cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);



// Verification
//Total sum of elements will be NxNxN
int count = 0;
    for(int i = 0; i < N; i++){
       for( int j = 0; j < N; j++){
         count = count + *(c + i*N + j);
    }
}

printf("Total sum of elements of c after matrix multiplication of a and b: %d\n", count);

//Deallocate Device  memory 
cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);

// Deallocate host memory
free(a); free(b); free(c);

}
