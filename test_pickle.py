# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 21:43:53 2021

@author: havre
"""
import numpy as np
from scipy import interpolate
# from scipy.integrate import solve_ivp

import pickle


# with open("D:/Tonstad/mypickle.pickle", 'rb') as f:
with open("D:/Tonstad/Q40_1s.pickle", 'rb') as f:
    tri = pickle.load(f)
 
    
#%%

import h5py

t_max = 1
steps = t_max * 20
    

with h5py.File("D:/Tonstad/Utvalde/Q40.hdf5", 'r') as f:
    Umx = np.array(f['Umx'])[0:steps,:]

    (I,J) = (int(np.array(f['I'])),int(np.array(f['J'])))

Umx_reshape = Umx.reshape((len(Umx),J,I))[:,1:114,1:125]

nonan = np.invert(np.isnan(Umx_reshape.ravel()))
    
Umx_lang = Umx_reshape.ravel()[nonan]


#%%
#For fleire punkt, slik det var i oppskrifta på internett

# uvw = (0,-88.5,87)
uvw = np.array([[0,-88.5,87],[0.03,-87,86.9]])

d=3
simplex = tri.find_simplex(uvw)
vertices = np.take(tri.simplices, simplex, axis=0)
temp = np.take(tri.transform, simplex, axis=0)
delta = uvw - temp[:, d]
bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))
            
interpolert_verdi = np.einsum('nj,nj->n', np.take(Umx_lang, vertices), wts)    
    
#%%
#For berre eitt punkt

uvw = np.array([0,-88.5,87])

d=3
simplex = tri.find_simplex(uvw)
vertices = np.take(tri.simplices, simplex, axis=0)
temp = np.take(tri.transform, simplex, axis=0)
delta = uvw - temp[d]
bary = np.einsum('jk,k->j', temp[:d, :], delta)
wts = np.hstack((bary, 1 - bary.sum(axis=0, keepdims=True)))
            
interpolert_verdi = np.einsum('j,j->', np.take(Umx_lang, vertices), wts) 
