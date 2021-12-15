# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
import matplotlib
matplotlib.use("Agg")
# from kornfordeling import get_PSD_part
import pickle
# import random
from sti_gen import Particle, Rib, simulering
from lag_video import sti_animasjon
# import numpy as np
# from datagenerering import lag_tre, tre_objekt, lag_tre_multi
# from hjelpefunksjonar import finn_fil
# import ray
import datetime
from pathlib import Path
from subprocess import run

talsamling = [10]
rnd_seed=1

pickle_fil = Path("../TONSTAD_TWO_Q20_TWO2.pickle")

t_span = (0,179)
print("Skal henta tre_objekt")
with open(pickle_fil,'rb') as f:
    tre = pickle.load(f)
print("Ferdig med tre-objektet")

ribs = [Rib(rib) for rib in tre.ribs]

sim_args = dict(fps = 20, t_span=t_span,
linear = True, lift = True, addedmass = True, wrap_max = 50,
method = 'BDF', atol = 1e-1, rtol = 1e-1, 
 verbose = False, collision_correction = True, multi = True)
laga_film = True


tider = {}
for tal in talsamling:
    talstart = datetime.datetime.now()

    partikkelfil = Path(f"particles_{pickle_fil.stem}_{sim_args['method']}_{tal}_{sim_args['atol']:.0e}.pickle")
    if not partikkelfil.exists():
        particle_list = simulering(tal, rnd_seed, tre, **sim_args)
        with open(partikkelfil, 'wb') as f:
            pickle.dump(particle_list, f)
    else:
        with open(partikkelfil, 'rb') as f:
            particle_list = pickle.load(f)

    if laga_film:
        start_film = datetime.datetime.now()
        film_fil = f"sti_{sim_args['method']}_{len(particle_list)}_{sim_args['atol']:.0e}.mp4"
        sti_animasjon(particle_list, ribs,t_span=t_span, hdf5_fil = pickle_fil.with_suffix(".hdf5"),  utfilnamn=film_fil, fps=sim_args['fps'])
        print("Brukte  {} s på å laga film".format(datetime.datetime.now() - start_film))

        run(f"rsync {film_fil} havrevol@login.ansatt.ntnu.no:", shell=True)


    tider[tal] = datetime.datetime.now() - talstart

print(tider)
