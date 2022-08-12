from random import randint
from time import sleep
from hjelpefunksjonar import f2t
from pathlib import Path
import pickle
import ray
skalering = 40

@ray.remote
def funksjon(tre,tal):
    soving = randint(30,80)
    print(tal, " skal sova ",soving)
    sleep(soving)
    
    return tal, soving

f_span = (0,3598)

t_span = (f2t(f_span[0],scale=skalering), f2t(f_span[0],scale=skalering))

pickle_fil = Path("./data/rib50_Q40_1_scale40.pickle")
hdf5_fil = pickle_fil.with_suffix(".hdf5")

with open(pickle_fil,'rb') as f:
    tre = pickle.load(f)

# ribs = [Rib(rib) for rib in tre.ribs]
print("her")
ray.init(local_mode=False)

tre_ray = ray.put(tre)

jobbar = [funksjon.remote(tre_ray, i) for i in range(20)]

svar = []
not_ready = jobbar
while True:
    ready, not_ready = ray.wait(not_ready)
    svaret = ray.get(ready[0])
    print("fekk til svar:",svaret)
    svar.append(svaret)
