#my_arr = np.arange(10000000) #may run out of memory
#my_list = list(range(10000000)
import sys
import numpy as np
import time

size = 1000000

my_arr = np.arange(size)
my_list = list(range(size))

# Testing a NumPy array
start = time.time() 
for _ in range(10): my_arr2 = my_arr * 2
end = time.time()
print ("Numpy Timing:", end-start)
# Testing a Python list

start = time.time()
for _ in range(10): my_list2 = [x * 2 for x in my_list]
end = time.time()
print ("Built-in Python Timing:", end-start)
