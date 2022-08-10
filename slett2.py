from datetime import datetime
from pathlib import Path
from time import sleep
from tre_server import tre_server
from tre_client import tre_client

from multiprocessing import Pool
import numpy as np

# tenar = Process(target=tre_server, daemon=True, args=(port,fname))
# tenar.start()

def hent(t):
    port = 5010
    tre = tre_client.get_tre_client(port)
    print(datetime.now(), "skal sova i ", t)
    sleep(t)
    print(datetime.now(), "ferdig å sova i ",t)
    tri, u = tre.get_tri_og_U(t)
    print(tri.min_bound)
    return tri.min_bound


t_ar = [7,3,15,26,9,2,47, 6,4,27,46,88,55,31]

with Pool(processes=7) as pool:
    result = pool.map(hent, t_ar)
    print(result)

