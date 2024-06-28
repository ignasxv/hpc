#include <stdio.h>

// Define matrix dimensions
#define N 10000
#define M 1000

// CUDA kernel to add two matrices (single CUDA thread)
__global__ void addMatrixSingleThread(float *a, float *b, float *c) {
    // Calculate linear index for 2D arrays
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Add corresponding elements
    c[idx] = a[idx] + b[idx];
}

// CUDA kernel to add two matrices (multiple CUDA threads)
__global__ void addMatrixMultipleThreads(float *a, float *b, float *c) {
    // Calculate linear index for 2D arrays
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = col + row * M;

    // Add corresponding elements
    c[idx] = a[idx] + b[idx];
}

int main() {
    // Host arrays
    float *h_a, *h_b, *h_c_single, *h_c_multiple;

    // Device arrays
    float *d_a, *d_b, *d_c_single, *d_c_multiple;

    // Calculate memory size
    size_t size = N * M * sizeof(float);

    // Allocate memory on host
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_single = (float*)malloc(size);
    h_c_multiple = (float*)malloc(size);

    // Allocate memory on device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_single, size);
    cudaMalloc(&d_c_multiple, size);

    // Initialize host arrays
    for (int i = 0; i < N * M; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Copy host data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    dim3 blockDim(32, 32); // 32x32 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch single-thread kernel
    addMatrixSingleThread<<<gridDim, blockDim>>>(d_a, d_b, d_c_single);

    // Synchronize device
    cudaDeviceSynchronize();

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_single;
    cudaEventElapsedTime(&milliseconds_single, start, stop);

    // Start timer for multiple-threads kernel
    cudaEventRecord(start);

    // Launch multiple-threads kernel
    addMatrixMultipleThreads<<<gridDim, blockDim>>>(d_a, d_b, d_c_multiple);

    // Synchronize device
    cudaDeviceSynchronize();

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds_multiple;
    cudaEventElapsedTime(&milliseconds_multiple, start, stop);

    // Copy results back to host
    cudaMemcpy(h_c_single, d_c_single, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_multiple, d_c_multiple, size, cudaMemcpyDeviceToHost);

    // Verify results (optional)

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_single);
    cudaFree(d_c_multiple);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c_single);
    free(h_c_multiple);

    printf("Time taken for single-thread kernel: %f ms\n", milliseconds_single);
    printf("Time taken for multiple-threads kernel: %f ms\n", milliseconds_multiple);

    return 0;
}

