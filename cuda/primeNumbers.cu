#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <ctime>
#include <cstdio>


#define UPPER 1131072
#define LOWER 2

__global__ void primes_in_range(int *result)
{
	int number = LOWER + (blockIdx.x * blockDim.x) + threadIdx.x;  //number can be (LOWER + 0) to (LOWER + total threads i.e # of blocks * # of threads/block) 
	if (number > UPPER)
	{
		return;                                                //thread won't execute when the number exceeds the upperbound
	}

        if (number == 2) atomicAdd(result, 1);                         //thread executes if number is 2 which is considered as a prime number
	if (number % 2 == 0) return;                                   //for even numbers, thread won't execute at all
	for (long divisor = 3; divisor < (number/2); divisor+=2)       //for odd numbers, thread executes only if it is not divisible 
	{
		if (number % divisor == 0)
		{
			return;
		}
	}

	atomicAdd(result, 1); //when a thread executes this operation, memory address is read, has the value of val (i.e.1) 
                              //is added to it, and the result is written back to memory.
}

int main()
{
	int begin = std::clock();

	int *result;
	cudaMallocManaged(&result, 4);
	*result = 0;

	primes_in_range<<<1105, 1024>>>(result); // # of blocks = ceil((UPPER-LOWER)/1024))
	cudaDeviceSynchronize();

//	int end = std::clock();
//	int duration = double(end - begin) / CLOCKS_PER_SEC * 1000;
	
	printf("Number of prime numbers between %d and %d is %d \n",LOWER,UPPER,*result);
	
//	getchar();
	return 0;
}
