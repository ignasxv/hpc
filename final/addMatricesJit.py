import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba import cuda
import numpy as np

@skip_on_cudasim("cudasim doesn't support cuda import at non-top-level")
class TestVecAdd(CUDATestCase):
    """
    Test simple vector addition
    """

    def setUp(self):
        # Prevent output from this test showing
        # up when running the test suite
        self._captured_stdout = captured_stdout()
        self._captured_stdout.__enter__()
        super().setUp()

    def tearDown(self):
        # No exception type, value, or traceback
        self._captured_stdout.__exit__(None, None, None)
        super().tearDown()

    def test_ex_vecadd(self):
        # ex_vecadd.kernel.begin
        @cuda.jit
        def f(a, b, c):
            # like threadIdx.x + (blockIdx.x * blockDim.x)
            tid = cuda.grid(1)   # tid => thread ID
            size = len(c)

            if tid < size:
                c[tid] = a[tid] + b[tid]
        # ex_vecadd.kernel.end

        N = 10000000
        a = np.arange(N)
        b = np.arange(N)
        c = np.zeros(N)

        # Launch the kernel
        threadsperblock = 256
        blockspergrid = (N + (threadsperblock - 1)) // threadsperblock
        f[blockspergrid, threadsperblock](a, b, c)

        # Verify the results
        expected_result = a + b
        np.testing.assert_array_equal(c, expected_result)

if __name__ == '__main__':
    unittest.main()

