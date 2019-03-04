import math
import timeit
import itertools as it
import numpy as np
import pandas as pd
import numba as nb
import numba.cuda as cuda

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

from timeit import default_timer as timer
import datetime
from dateutil.tz import gettz
tz = gettz("US/Central")
def time_stamp():
    t = datetime.datetime.now(tz)
    return f"{t.strftime('%Y-%m-%d %H:%M:%S')} {tz._ttinfo_dst.abbr}"

print(f"Start time = {time_stamp()}")

tpb = cuda.get_current_device().MAX_THREADS_PER_BLOCK
blk_shp = (tpb, 1)
grd_shp = (2**10, 2**8)
ary_shp = grd_shp + blk_shp

typ = np.uint16
lower = np.sqrt(np.iinfo(typ).min)  #sqrt so all entries of A*B stay in range
upper = np.sqrt(np.iinfo(typ).max)
ran = upper - lower

rnd = np.random.RandomState(42)  # Set random seed so we get the same answer every time
X = rnd.uniform(size=ary_shp)  #U[0,1)
Y = rnd.uniform(size=ary_shp)  #U[0,1)

A = (X * ran + lower).astype(typ)  #convert to ran and typ
B = (Y * ran + lower).astype(typ)  #convert to ran and typ
# print(lower, A.min())
# print(upper, A.max())


A_gpu = cuda.to_device(A)
B_gpu = cuda.to_device(B)
C_gpu = cuda.device_array_like(A_gpu)

def dot1():
    @cuda.jit
    def ker(A, B, C):
        bx = cuda.blockIdx.x
        by = cuda.blockIdx.y
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        C[bx, by, tx, ty] = A[bx, by, tx, ty] * B[bx, by, tx, ty]

    ker[grd_shp, blk_shp](A_gpu, B_gpu, C_gpu)
    C = C_gpu.copy_to_host()
    gpu_sol = np.sum(C)
    rel_err = np.abs((gpu_sol-cpu_sol) / cpu_sol)
    assert rel_err < 0.01, f"cpu_sol = {cpu_sol}, gpu_sol = {gpu_sol}, rel_err = {rel_err}"

cpu_sol = np.sum(A*B)
print("Timing the CPU")
cpu_time = timeit.timeit('np.sum(A*B)', setup='import numpy as np', number=3)
print("Timing my GPU")
gpu_time = timeit.timeit('dot1()', number=3)
