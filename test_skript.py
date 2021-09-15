# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""

import numpy as np
from sti_gen import get_u, Rib, Particle, sti_animasjon, lag_sti, particle_copy
from datagenerering import hent_tre, lag_tre, tre_objekt, lag_tre_multi
from hjelpefunksjonar import finn_fil
import ray
ray.init() 
import random
import matplotlib.pyplot as plt
import pickle

# #%% Førebu

tre_fil = "../tre_0_60_mednullribbe.pickle"
#tre_fil = finn_fil(["C:/Users/havrevol/Q40_60s.pickle", "D:/Tonstad/Q40_60s.pickle", "../Q40_60s.pickle"])

t_span = (0,19)

# %timeit get_u(random.uniform(0,20), [random.uniform(-88,88), random.uniform(-70,88)], tri, ckdtre, U, linear=True)

# ribs = [Rib((-61.07,-8.816),50.2,7.8), Rib((39.03,-7.53), 50, 7.8), Rib((-100,-74.3), 200, -10)]
# ribs = [Rib([ [-10.83594, -1.1020], [-11.0196, -8.9075], [-60.79146, -0.8265], [-60.97512, -8.6320] ]), 
#         Rib([ [89.0751, 0.09183], [89.0751, -7.71372], [39.02775, 0.09183], [39.02775, -7.71372] ]),
#         Rib([[93.29928, -74.84145], [-93.20745, -72.63753], [-93.20745, -98.80908], [93.29928,- 98.80908]])]



#%%

# #%% Test løysing av difflikning
# svar_profft = solve_ivp(stein.f,(0.375,0.4), np.array([-88.5,87,0,0]), args=(tri, U))
# svar_profft2 = rk_3(stein.f, (0.375,0.4), np.array([-88.5,87,0,0]))
# svar = rk_2(stein.f, np.array([-88.5,87,0,0]), (0,0.4), 0.01, tri, U)

# #%% Test kollisjon
# stein2 = Particle([-80,50],3)
# koll = stein2.checkCollision([-63,-1], ribs[0]) #R2
# koll2 = stein2.checkCollision([-40,-1], ribs[0]) #R3 (midten av flata)

particle_list = [Particle(0.05, [-80,85,0,0]), Particle(0.1, [-80,80,0,0]), Particle(0.2, [-80,75,0,0]) ]

# #%%

# try: tri
# except NameError: tri = None
# if tri is None:
tre = tre_objekt(tre_fil, t_span)

tre_plasma = ray.put(tre)

ribs = [Rib(rib) for rib in tre.ribs]
    
linear = True
lift = True
addedmass = True

f_args = (tre_plasma, linear, lift, addedmass)
solver_args = {'atol': 1e-3, 'rtol':1e-1, 'method':'RK45', 'args':f_args}

fig, ax = plt.subplots()

tols = [(1e-3,1e-1), (1e-2,1e-1), (1e-1,1e-1) ]

part0 = particle_list[0]



methods = ['RK45', 'RK23',  'Radau', 'BDF', 'LSODA'] # tok ut DOP853, for den tok for lang tid.

kombinasjon = []

for tol in tols:
    for sol in methods:
        kombinasjon.append({'tol':tol,'sol':sol, 'pa':particle_copy(part0) } )
  

stiar = []

import time

result_ids = []

for k in kombinasjon:
    f_args = (k['pa'], tre_plasma, linear, lift, addedmass)
    solver_args = {'atol': k['tol'][0], 'rtol':k['tol'][1], 'method':k['sol'], 'args':f_args}
    k['id'] = lag_sti.remote(ribs, t_span, solver_args=solver_args, wraparound=True)
    

for k in kombinasjon:
    k['pa'].sti = ray.get(k['id'])
    stiar.append(k['pa'].sti)
    
    # print("Ferdig med ", pa.init_position)
    # print("Den som har diameter ", pa.diameter)
    # print("solver er", k[1])
    # print("atol er", solver_args['atol'])
    # print("rtol er", solver_args['rtol'])
    # ax.plot(pa.sti[:,1],pa.sti[:,2])
        
with open("stiar.pickle", 'wb') as f:
    pickle.dump(k, f)


# particle_pool = multiprocessing.Pool()

# particle_result = particle_pool.map(pool_helper, particle_list)


# sti_animasjon(particle_list,t_span=t_span)

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

