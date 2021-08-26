import ray
# import multiprocessing
import numpy as np
import scipy.spatial.qhull as qhull
from datagenerering import tre_objekt
import random
import time
ray.init()  

@ray.remote
def func(tre, tx):
    i = tx[0]
    tx = tx[1:]
    start = time.time()
    tri_einskild = tre.get_tri(tx)
    simplex = tri_einskild.find_simplex(tx)
    end = time.time()
    print("Nummer ",i," brukte ",end-start," sekund på å bli henta.")
    return simplex

tre_fil = "../Q40_60s.pickle"
t_span = (0,60)

start = time.time()
tre = tre_objekt(tre_fil, t_span)
end = time.time()
print("Treet brukte ",end-start," sekund på å bli laga.")

print("Skal putta treet i array_id")
array_id = ray.put(tre)
print("Ferdig med det")

random_positions = [(i, random.uniform(t_span[0],t_span[1]), random.uniform(-88,88), random.uniform(-70,88)) for i in range(200) ]

print("Byrjar den spennande multi-delen")
result_ids = [func.remote(array_id, i) for i in random_positions]
print("Ferdig med result_ids")
output = ray.get(result_ids)

print("Ferdig!")
