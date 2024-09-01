from memory_profiler import profile
import sys


@profile
def test_mem(x: int):
    return x

if sys.argv[0]:
    iters = int(sys.argv[1])
else:
    iters = 1000000

x = 0

for i in range(iters):
    x = test_mem(i)
