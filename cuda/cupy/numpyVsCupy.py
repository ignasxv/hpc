import numpy as np
import cupy as cp
import time


start=time.time()
#x_cpu=np.array([1,2,3,4,5,6,7,8,9,10])
#x_cpu=np.arange(1,10, dtype=float)

### Numpy and CPU
s = time.time()
x_cpu = np.ones((1000,1000,1000))
x_cpu *= 5000
x_cpu *= x_cpu
x_cpu += x_cpu
e = time.time()
print("Numpy performance")
print(e - s)
print("\n")

### CuPy and GPU
s = time.time()
x_gpu = cp.ones((1000,1000,1000))
x_gpu *= 5000
x_gpu *= x_gpu
x_gpu += x_gpu
cp.cuda.Stream.null.synchronize()
e = time.time()
print("Cupy performance")
print(e - s)
