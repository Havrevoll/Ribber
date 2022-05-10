from collections import defaultdict
import pickle
from pathlib import Path
import re
import numpy as np
from scipy.interpolate import interp1d
from math import sqrt

# diameter_intervall = np.array([[0.05, 0.06,0.07, 0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.],
# [0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12.]])
# fartar = 1.65*9810* diameter_intervall ** 2 / (20*1.5674+np.sqrt(0.75*1.1*1.65*9810*diameter_intervall**3))
# skalert_fart = fartar*np.sqrt(20)

# d = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,9000, 10000])
# u = 1.65*9810 * d ** 2 / (20*1.5674+np.sqrt(0.75*1.1*1.65*9810*d**3))
# f = interp1d(u,d)

# skalert_d = f(skalert_fart)

# Q = np.array([20,40,60,80,100,120,140])
# L = np.array([25,50,75])

# stabla =  np.vstack((diameter_intervall,fartar,skalert_fart,skalert_d))

# alle = np.empty((len(L), len(Q), len(stabla), len(stabla[0])))


# nested_dict = lambda: defaultdict(nested_dict)
# alle = nested_dict()


import ray

@ray.remote
def oppsummer(fil):
    samling = []

    if True:
        print(fil.name)
        fil_L = re.search("rib(\d\d)", fil.name).group(1)
        fil_Q = re.search("Q(\d?\d\d)", fil.name).group(1)
        fil_nummer = re.search("(\d)_BDF", fil.name).group(1)
        fil_sim_nummer = re.search("_(\d?)\.pickle", fil.name)
        if fil_sim_nummer:
            fil_sim_nummer = fil_sim_nummer.group(1)
        else:
            fil_sim_nummer = 1

        assert fil.exists()

        caught =0
        uncaught = 0
        with open(fil,'rb') as fila:
            pa = pickle.load(fila)
        del fila

        # samling.append(

        for p in pa:
            try:
                caught = p.sti_dict[int(round(p.sti_dict['final_time']*100))]['caught']
                velocity = sqrt(p.sti_dict[int(round(p.sti_dict['final_time']*100))]['position'][2]**2 + p.sti_dict[int(round(p.sti_dict['final_time']*100))]['position'][3]**2 )
                if not caught and velocity < 0.001:
                    caught = True
                samling.append(dict(L = fil_L, Q = fil_Q, forsøk=fil_nummer, simulering = fil_sim_nummer,d=p.diameter,caught=caught))
            except Exception as e:
                raise Exception(f"partikkel nr {p.index}").with_traceback(e.__traceback__)
    return samling

if __name__ == '__main__':
    ray.init()
    filbane=Path("partikkelsimulasjonar/samla") 
    utfil=Path('./partiklar.pickle')
    samling = []
    jobbar = []

    jobbar = [oppsummer.remote(fil) for fil in filbane.glob('*.pickle')]
    not_ready = jobbar
    while True:
        ready, not_ready = ray.wait(not_ready)
        samling.extend(ray.get(ready[0]))
        if len(not_ready)==0:
            break

    with open(utfil,'wb') as fila:
        pickle.dump(samling, fila)