# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
import matplotlib
matplotlib.use("Agg")
import pickle
from sti_gen import Rib, simulering #,Particle
from lag_video import sti_animasjon
from hjelpefunksjonar import finn_fil
import datetime
from pathlib import Path
import numpy as np

tal = 1000
rnd_seed=1
tider = {}

talstart = datetime.datetime.now()
t_span = (0,100)

sim_args = dict(t_span=t_span,
    wrap_max = 0,
    method = 'BDF',
    verbose = False, multi = False)
laga_film = True

partikkelfil = Path(f"./partikkelsimulasjonar/simple.pickle")
if not partikkelfil.exists():

    particle_list = simulering(tal, rnd_seed, **sim_args)
    with open(partikkelfil, 'wb') as f:
        pickle.dump(particle_list, f)

else:
    with open(partikkelfil, 'rb') as f:
        particle_list = pickle.load(f)

if laga_film:
        film_fil = partikkelfil.with_suffix(".mp4") #Path(f"./filmar/sti_{pickle_fil.stem}_{sim_args['method']}_{len(particle_list)}_{sim_args['atol']:.0e}.mp4")
        # if not film_fil.exists():
        ribs =  []
        ribs.append(Rib([[-1.,0.],[-1.,-1.],[0.,-1.],[0.,0.]]))
        ribs.append(Rib([[15.,0.],[15.,-1.],[16.,-1.],[16.,0.]]))
        ribs.append(Rib([[-1.,0.],[16.,0.],[16.,-1],[-1.,-1.]]))
        sti_animasjon(particle_list, ribs,t_span=t_span, utfilnamn=film_fil, fps=60)
