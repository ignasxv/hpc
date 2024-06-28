import numba  #  open source JIT compiler that translates a subset of Python and NumPy code into fast machine code at run time
import numpy
import time
 
@numba.jit    # decorators: wrapper to existing functions which extend them

def sum(x):
    total = 0
    for i in range(x.shape[0]):
        total +=x[i]
    return total
 
x = numpy.arange(10000000);

t0 = time.time() 
sum(x)
t1 = time.time()
total = t1 - t0
print(total)
