# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""

import numpy as np
from sti_gen import get_u, Rib, Particle, sti_animasjon
from datagenerering import get_velocity_data, hent_tre, lag_tre
import random

# #%% Førebu

# %timeit get_u(random.uniform(0,20), [random.uniform(-88,88), random.uniform(-70,88)], tri, ckdtre, U, linear=True)
try: U
except NameError: U = None

if U is None:
    # U  = get_velocity_data(20)
    # dudt_mean = np.nanmean(get_velocity_data(20, one_dimensional=False)[2:],1)
    # tri = hent_tre()
    # ckdtre = lag_tre(t_span=(0,20))
    linear = True


ribs = [Rib((-61.07,-8.816),50.2,7.8), 
        Rib((39.03,-7.53), 50, 7.8), 
        Rib((-100,-74.3), 200, -10)]


trees = {}

import time

     
   
for i in range(5*10):
    start = time.time()   
    trees[i] = lag_tre((i/10, (i+1.5)/10), nearest=False)

    end = time.time()
    print("i = ", i)
    print("Det tok så mange sekund å laga dette treet:", end - start)

del start,end

#%%
from datagenerering import lagra_tre

lagra_tre(trees, "C:/Users/havrevol/trees_0-5s.pickle")

# #%% Test løysing av difflikning
# svar_profft = solve_ivp(stein.f,(0.375,0.4), np.array([-88.5,87,0,0]), args=(tri, U))
# svar_profft2 = rk_3(stein.f, (0.375,0.4), np.array([-88.5,87,0,0]))
# svar = rk_2(stein.f, np.array([-88.5,87,0,0]), (0,0.4), 0.01, tri, U)

# #%% Test kollisjon
# stein2 = Particle([-80,50],3)
# koll = stein2.checkCollision([-63,-1], ribs[0]) #R2
# koll2 = stein2.checkCollision([-40,-1], ribs[0]) #R3 (midten av flata)



# #%%
# t_max = 15
# tol = (1e-4,1e-2)
# pa = Particle(0.05)
# pa.sti = pa.lag_sti([-80,85,0,0], (0,t_max), args=(tri, ckdtre, U, linear), wraparound=True, atol=tol[0], rtol=tol[1])

# sti_animasjon([pa],t_span=(0,t_max))

# #%%
# get_u.counter = 0

# def test_part(t_max = 15):
#     # methods = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
#     # sizes = [0.07,0.06,0.05,0.04,0.03,0.02]
#     # atol=1e-6, rtol=1e-3
#     tols = [(1e-6,1e-3), (1e-5,1e-3), (1e-5,1e-2), (1e-4,1e-2)]
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

