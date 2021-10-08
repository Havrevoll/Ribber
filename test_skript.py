# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
import matplotlib
matplotlib.use("Agg")
from kornfordeling import get_PSD_part
import pickle
import random
from sti_gen import Particle, simulering
import numpy as np
from datagenerering import hent_tre, lag_tre, tre_objekt, lag_tre_multi
from hjelpefunksjonar import finn_fil
import ray
import datetime

talsamling = [1000]
rnd_seed=1

tre_fil = finn_fil(["../Q40_0_60.pickle", "../Q40_0_10.pickle"])
t_span = (0,59)

sim_args = dict(fps = 20, t_span=t_span,
linear = False, lift = False, addedmass = False, wraparound = False,
method = 'RK23', atol = 1e-1, rtol = 1e-1, 
laga_film = False, verbose = False, collision_correction = False)
print("Skal byrja å byggja tre_objekt")
tre = tre_objekt(tre_fil, t_span)

tider = {}

for tal in talsamling:
    talstart = datetime.datetime.now()
    partikkelfil = f"particles_{sim_args['method']}_{tal}_{sim_args['atol']:.0e}_noaddedmass_ray.pickle"

    particle_list = simulering(tal, rnd_seed, tre, **sim_args)

    with open(partikkelfil, 'wb') as f:
        pickle.dump(particle_list, f)

    tider[tal] = datetime.datetime.now() - talstart

print(tider)
