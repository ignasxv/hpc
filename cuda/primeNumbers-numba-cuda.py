import numpy as np
from numba import jit, cuda
import math
import time

@cuda.jit()
def seive(array):
    
    i = cuda.grid(1)
    k = 2
    while k <= math.sqrt(array[i]):
        if array[i] % k == 0:
            array[i] = 0
        #else:
             #array[i] = 1
            break
        #array[i] = 1
        k += 1
     

size = 1131072  # create 
#arr = np.array([size], np.int32) 
arr = np.array([i for i in range(0, size)], np.int32)  # initialized array with size and data type as int32 to pass to cuda kernel function
total = 0; #np.zeros((1), np.int32)
#result = 0
#for i in range(size):
#    result+=arr[i]
#print(arr)
start_time = time.time()
seive[1105, 1024](arr)    # each unique thread to check the primality of each number i.e. ceil (size/1024)
#print(arr)
for i in range(size):
    if arr[i] != 0:
       arr[i] = 1
       total+=arr[i]
#print(total)
elapsed_time = time.time() - start_time
print("Number of Prime Numbers = ",total-1)
print("Execution Time = ", elapsed_time)
