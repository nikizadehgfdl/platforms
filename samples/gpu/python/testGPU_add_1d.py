# (minienv) [Niki.Zadeh@lscamd50-d python]$ python testGPU_add_1d.py
# numba vectorize cuda  time=   0.19 secs, sum=   100001000000.0
# numba vectorize cpu   time=  89.00 secs, sum=   100001000000.0
# numba jit             time=  90.25 secs, sum=   100001000000.0
# numpy                 time=  95.07 secs, sum=   100001000000.0

import numpy as np
from numba import vectorize,jit
from timeit import default_timer as timer

def Add1(a, b):
  c=a
  for j in range(100000):  
    c=c+b
  return c

@jit(nopython=True)
def Add1_jit(a, b):
  c=a
  for j in range(100000):  
    c=c+b
  return c

@vectorize(['float64(float64, float64)'], target='cuda')
def Add1_cuda(a, b):
  c=a
  for j in range(100000):  
    c=c+b
  return c

@vectorize(['float64(float64, float64)'], target='cpu')
def Add1_cpu(a, b):
  c=a
  for j in range(100000):  
    c=c+b
  return c

# Initialize arrays
N = 1000000
A = np.ones(N, dtype=np.float64)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)

# Add arrays on GPU
start = timer()
C = Add1_cuda(A, B)
print("numba vectorize cuda  time=%7.2f secs, sum=  " % float(timer()-start), C.sum() )

start = timer()
C = Add1_cpu(A, B)
print("numba vectorize cpu   time=%7.2f secs, sum=  " % float(timer()-start), C.sum() )

start = timer()
C = Add1_jit(A, B)
print("numba jit             time=%7.2f secs, sum=  " % float(timer()-start), C.sum())

start = timer()
C = Add1(A, B)
print("numpy                 time=%7.2f secs, sum=  " % float(timer()-start), C.sum())

