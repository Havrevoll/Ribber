# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""

import numpy as np
from scipy.sparse import dia
from scipy.sparse.construct import rand
from sti_gen import get_u, Rib, Particle, sti_animasjon, lag_sti, particle_copy
from datagenerering import hent_tre, lag_tre, tre_objekt, lag_tre_multi
from kornfordeling import get_PSD_part
from hjelpefunksjonar import finn_fil
import ray
ray.init() 
import random
import matplotlib.pyplot as plt
import pickle

# #%% Førebu

tre_fil = "../tre_0_60.pickle"
#tre_fil = finn_fil(["C:/Users/havrevol/Q40_60s.pickle", "D:/Tonstad/Q40_60s.pickle", "../Q40_60s.pickle"])

t_span = (0,59)
tal = 15
linear, lift, addedmass = True, True, True
wraparound = False
atol, rtol = 1e-1, 1e-1
method = 'RK23'

# %timeit get_u(random.uniform(0,20), [random.uniform(-88,88), random.uniform(-70,88)], tri, ckdtre, U, linear=True)

diameters = get_PSD_part(tal)

# diameters = [0.08]
particle_list = [Particle(d, [-90,random.uniform(0,90)], random.uniform(0,50)) for d in diameters]

for p in particle_list:
    p.atol , p.rtol = atol, rtol
    p.method = method
    p.linear, p.lift, p.addedmass = linear, lift, addedmass
# part = Particle(0.08, [-80, 50,0,0])
    # random.uniform(0,88)

# particle_list = [Particle(0.05, [-80, 85,0,0]), Particle(0.1, [-80,80,0,0]), Particle(0.2, [-80,75,0,0]) ]

tre = tre_objekt(tre_fil, t_span)
print("Har laga tre_objekt, skal putta")
tre_plasma = ray.put(tre)
print("Har putta")

ribs = [Rib(rib) for rib in tre.ribs]
    

# atols = [1e-3, 1e-2, 1e-1 ]
# rtols = [1e-1, 1e-2] 
# methods = ['RK45', 'RK23'] # ,  'Radau', 'BDF', 'LSODA'] # tok ut DOP853, for den tok for lang tid.

# kombinasjon = []

# for atol in atols:
#     for rtol in rtols:
#         for sol in methods:
#             kombinasjon.append({'atol':atol,'rtol':rtol, 'sol':sol, 'pa':particle_copy(part)} )
            

# kombinasjon.append({'atol':1e-6,'rtol':1e-3, 'sol':'RK45', 'pa':particle_copy(part)} )
jobbar = []

for pa in particle_list:
# for ko in kombinasjon:
        solver_args = dict(atol=1e-1, rtol=1e-1, method='RK23', linear=linear, lift=lift, addedmass=addedmass, pa=pa, tre_plasma=tre_plasma)
        pa.job_id(lag_sti.remote(ribs, t_span, pa=pa, tre_plasma=tre_plasma, wraparound=True))
        
        


stiar = []
for jobb, part in zip(jobbar, particle_list):
    part.sti = ray.get(jobb)
    stiar.append(part.sti)


# for k in kombinasjon:
#     f_args = (k['pa'], tre_plasma, linear, lift, addedmass)
#     solver_args = {'atol': k['tol'][0], 'rtol':k['tol'][1], 'method':k['sol'], 'args':f_args}
#     k['id'] = lag_sti.remote(ribs, t_span, solver_args=solver_args, wraparound=True)
    
   
with open("sti.pickle", 'wb') as f:
    pickle.dump(stiar, f)

sti_animasjon(particle_list,t_span=t_span, utfilnamn="sti_RK45_og_23.mp4")

# #%%
# get_u.counter = 0

# def test_part(t_max = 15):
#     # methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
#     # sizes = [0.07,0.06,0.05,0.04,0.03,0.02]
#     # atol=1e-6, rtol=1e-3
#     tols = [(1e-6,1e-3), (1e-5,1e-3), (1e-5,1e-2), (1e-4,1e-2), (1e-3,1e-1)]
#     steinar = []
#     import time
    
#     for tol in tols:
#         start  = time.time()
#         pa = Particle(0.05)
#         try:
#             pa.sti = pa.lag_sti([-80,85,0,0], (0,t_max), args=(tri, ckdtre, U, linear), wraparound=True, atol=tol[0], rtol=tol[1])
#         except Exception:
#             print("må gå vidare")
    
#         end = time.time()            
#         print("toleransar: ", tol," iterasjonar: ", get_u.counter, "Det tok", end-start)
#         get_u.counter=0
#         steinar.append(pa)
        
#     sti_animasjon(steinar,t_max)
# test_part()


# #%%
# stein = Particle(1)
# stein2 = Particle(0.1) 
# stein3 = Particle(0.05)
# stein4 = Particle(0.02)

# stein5 = Particle(0.2)

# t_max = 15
# tol = (1e-4,1e-2)

# args = {'t_span':(0,t_max), 'args':(tri, ckdtre, U, linear), 'wraparound':True, 'atol':tol[0], 'rtol':tol[1]}

# stein.sti = stein.lag_sti([-88,90,0,0], **args )
# print(get_u.counter)
# get_u.counter=0
# stein3.sti = stein3.lag_sti([-88,80,0,0],**args)
# print(get_u.counter)
# get_u.counter=0
# stein2.sti = stein2.lag_sti([-88,70,0,0],**args)
# print(get_u.counter)
# get_u.counter=0
# stein4.sti = stein4.lag_sti([-88,60,0,0],**args)
# print(get_u.counter)
# get_u.counter=0
# stein5.sti = stein5.lag_sti([-90, random.uniform(-60,88),0,0], **args)
# print(get_u.counter)
# get_u.counter=0

# sti_animasjon([stein, stein2, stein3,stein4,stein5],t_max=t_max)

