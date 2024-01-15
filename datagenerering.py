# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:37:35 2021

@author: havrevol
"""
# import h5py
import pickle
import numpy as np

# from hjelpefunksjonar import ranges, finn_fil
# import datetime
# import re

# filnamn = "../two_q40.hdf5" #finn_fil(["D:/Tonstad/utvalde/Q40.hdf5", "C:/Users/havrevol/Q40.hdf5", "D:/Tonstad/Q40.hdf5"])

# pickle_fil = finn_fil(["../Q40_60s.pickle", "D:/Tonstad/Q40_20s.pickle", "C:/Users/havrevol/Q40_20s.pickle", "D:/Tonstad/Q40_2s.pickle"])

# print("pickle fil er ", pickle_fil) 

def lagra_tre(tre, fil):
    with open(fil, 'wb') as f:
        pickle.dump(tre, f)

# def hent_tre(fil=pickle_fil):
#     with open(fil, 'rb') as f:
#         tri = pickle.load(f)
 
#     return tri

class tre_objekt:
    def __init__(self, kdtre, U_kd):
        """Eit objekt med relevant informasjon

        Args:
            kdtre (_type_): _description_
            U_kd (_type_): _description_
        """
        self.kdtre = kdtre
        self.U_kd = U_kd
    
    def get_kd_U(self, tx):
        # while True:
        #         try:
        #             self.U_kd[:,self.kdtre.query(np.swapaxes(tx, -2,-1))[1]]
        #             break
        #         except IndexError:
        #             tx[np.abs(tx)>1e100] /= 1e10
                
        return self.U_kd[:,self.kdtre.query(np.swapaxes(tx, -2,-1))[1]] #, nullfart, np.zeros((2,2))
    
# def get_vel_snippets(t_span):
#     t_min = t_span[0]
#     t_max = t_span[1]
    
#     velocities= {}
    
#     for i in range(t_min*10,t_max*10):
#         i_span = (i/10,(i+1.5)/10)
        
     
        
#         velocities[i] = get_velocity_data(i_span)
        
#     return velocities


# import scipy.interpolate as spint

# def interp_weights(xyz, uvw):
    
#     d=3
    
#     tri = qhull.Delaunay(xyz)
#     simplex = tri.find_simplex(uvw)
#     vertices = np.take(tri.simplices, simplex, axis=0)
#     temp = np.take(tri.transform, simplex, axis=0)
#     delta = uvw - temp[:, d]
#     bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
#     return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

# def interpolate(values, vtx, wts):
#     return np.einsum('nj,nj->n', np.take(values, vtx), wts)

# Her er interpoleringa for qhull, lineært, altså.
    # uvw = (0,-88.5,87)
    # simplex = tri.find_simplex(uvw)
    # vertices = np.take(tri.simplices, simplex, axis=0)
    # temp = np.take(tri.transform, simplex, axis=0)
    # delta = uvw - temp[3, :]
    # bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    # wts = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True))
                    
    # interpolate = np.einsum('nj,nj->n', np.take(values, vtx), wts)    

    # Metoden er den same som denne:
    # interpolate.griddata((t_lang, x_lang, y_lang), Umx_lang, uvw[0], method='linear')
    
    # Her er interpoleringa for CKD-tre, nearest neighbor, altså.
    # Umx_lang[tree.query(uvw)[1]]
    # dist, i = tree.query(uvw)
    
    # tree.U = U
    
