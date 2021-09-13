# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:37:35 2021

@author: havrevol
"""
import h5py
import pickle

import numpy as np

from hjelpefunksjonar import ranges, finn_fil

import scipy.spatial.qhull as qhull
from scipy.spatial import cKDTree

import multiprocessing

fps = 20

filnamn = "../Q40.hdf5" #finn_fil(["D:/Tonstad/utvalde/Q40.hdf5", "C:/Users/havrevol/Q40.hdf5", "D:/Tonstad/Q40.hdf5"])

pickle_fil = finn_fil(["../Q40_60s.pickle", "D:/Tonstad/Q40_20s.pickle", "C:/Users/havrevol/Q40_20s.pickle", "D:/Tonstad/Q40_2s.pickle"])
# print("pickle fil er ", pickle_fil) 

def lag_tre_multi(t_span, filnamn_ut=None):
    
    a_pool = multiprocessing.Pool()
    
    t_min = t_span[0]
    t_max = t_span[1]
    
    i = [(i/10,(i+1.5)/10) for i in range(int(t_min)*10,int(t_max)*10)]
    
    result = a_pool.map(lag_tre, i)
    
    i_0 =  range(int(t_min)*10,int(t_max)*10)
    
    trees = dict(zip(i_0, result))
            
    if filnamn_ut is None:
        return trees
    else:
        lagra_tre(trees, filnamn_ut)


def lag_tre(t_span=(0,1), nearest=False, kutt=False, inkluder_ribs = False):
    '''
    Ein metode som tek inn eit datasett og gjer alle reshapings-tinga for x og y, u og v og Re.

    '''
     
    t_min = t_span[0]
    t_max = t_span[1]
    
    piv_range = ranges(kutt)
    
    
    with h5py.File(filnamn, 'r') as f:
        Umx = np.array(f['Umx'])[int(t_min*fps):int(t_max*fps),:]
        Vmx = np.array(f['Vmx'])[int(t_min*fps):int(t_max*fps),:]
        (I,J) = (int(np.array(f['I'])),int(np.array(f['J'])))
        x = np.array(f['x']).reshape(J,I)[piv_range]
        y = np.array(f['y']).reshape(J,I)[piv_range]
        ribs = np.array(f['ribs'])
    
    
    Umx_reshape = Umx.reshape((len(Umx),J,I))[:,piv_range[0],piv_range[1]]
    Vmx_reshape = Vmx.reshape((len(Vmx),J,I))[:,piv_range[0],piv_range[1]]
    
    axis0 = np.take_along_axis(ribs[:,:,0], np.argpartition(ribs[:,:,1],-2),1)[:,-2:].T
    axis1= np.take_along_axis(ribs[:,:,1], np.argpartition(ribs[:,:,1],-2),1)[:,-2:].T
    for rib in np.stack((axis0,axis1),axis=0).swapaxes(0,2):
        

    # Legg inn automatisk henting av desse verdiane?
    x0 = -60.79
    x1 = -10.84
    y0 = -.8265
    y1 = -1.1020
    
    # y[64,18:55] = -.8265+ (x[0,18:55] + 60.79)* (-1.1020+0.8265)/(-10.84+60.79)
    y[64,18:55] = y0 + (x[0,18:55] - x0)* (y1 - y0)/(x1 - x0)
    
    # y[63,0:54]=-1.01
    Umx_reshape[:,64:70,0:55]=0
    Vmx_reshape[:,64:70,0:55]=0
    
    
    Umx_reshape[:,62:68,87:]=0
    Vmx_reshape[:,62:68,87:]=0
    Umx_reshape[:,68:70,89:]=0
    Vmx_reshape[:,68:70,89:]=0
    
    
    # if (nearest):
    #     dx = 1.4692770000000053 # eigentleg 91.83 * 16 * 1e-3 = 1.46928
    #     dy = 1.4692770000000053
    #     dt = 1/fps
        
    #     dudt = np.gradient(Umx_reshape,dt,axis=0)+Umx_reshape*np.gradient(Umx_reshape,dx,axis=2)+Vmx_reshape*np.gradient(Umx_reshape,dy,axis=1)
    #     dvdt = np.gradient(Vmx_reshape,dt,axis=0)+Umx_reshape*np.gradient(Vmx_reshape,dx,axis=2)+Vmx_reshape*np.gradient(Vmx_reshape,dy,axis=1)
        
    #     if (one_dimensional):
    #         dudt_lang = dudt.ravel()
    #         dvdt_lang = dvdt.ravel()
            
    
    # u_bar = np.nanmean(Umx,0)
    # v_bar = np.nanmean(Vmx,0)

    # u_reshape = u_bar.reshape((J,I))[1:114,1:125]
    # v_reshape = v_bar.reshape((J,I))[1:114,1:125]
    
    # if (one_dimensional):
    Umx_lang = Umx_reshape.ravel()
    Vmx_lang = Vmx_reshape.ravel()
        
    nonan = np.invert(np.isnan(Umx_lang))
        
    #     if (with_gradient):
    #         return np.array([Umx_lang[nonan], Vmx_lang[nonan], dudt_lang[nonan], dvdt_lang[nonan]])
    #     else: 
    #         return np.array([Umx_lang[nonan], Vmx_lang[nonan]])
    # else:
    #     return np.array([Umx_reshape, Vmx_reshape, dudt, dvdt])
    
    U = np.array([Umx_lang[nonan], Vmx_lang[nonan]])
    
    # U = get_velocity_data(t_span=t_span, with_gradient=nearest, one_dimensional = True)
    
    
    # Umx_reshape = U[0]
    
    # x = np.array(dataset['x']).reshape(J,I)[piv_range]
    # y = np.array(dataset['y']).reshape(J,I)[piv_range]
    
    t_3d,y_3d,x_3d = np.meshgrid(np.arange(t_min*fps,t_max*fps)/fps, y[:,0], x[0,:], indexing='ij')
    t_lang = t_3d.ravel()[nonan]
    x_lang = x_3d.ravel()[nonan]
    y_lang = y_3d.ravel()[nonan]
        
    txy = np.vstack((t_lang,x_lang,y_lang)).T
    
    # U = [i.ravel()[nonan] for i in U]
    
    # return txy, U
    
    # txy, U = get_txy(t_span, dataset = h5py.File(filnamn, 'r'), nearest=nearest)

    if (nearest):
        tree = cKDTree(txy)
    else:
        tree = qhull.Delaunay(txy)
    
    if (inkluder_ribs):
        return tree, U, ribs
    else:
        return tree, U



# def get_velocity_data(t_span=(0,1), with_gradient = False, one_dimensional = True):
#     t_min = t_span[0]
#     t_max = t_span[1]
#     steps = t_max * fps
    
#     piv_range = ranges()
    
# def get_txy(t_span=(0,1), dataset = h5py.File(filnamn, 'r'), nearest = False):


def lagra_tre(tre, fil):
    with open(fil, 'wb') as f:
        pickle.dump(tre, f)

def hent_tre(fil=pickle_fil):
    with open(fil, 'rb') as f:
        tri = pickle.load(f)
 
    return tri

class tre_objekt:
    def __init__(self, picklenamn, t_span):
        with open(picklenamn, 'rb') as f:
            self.tre = pickle.load(f)
            self.kdtre, self.U_kd, self.ribs = lag_tre(t_span=t_span, nearest=True, inkluder_ribs=True)
    
    
    # def find_simplex(self, x):
    #     t = x[0]
    #     i = int(t*10)
        
        # delaunay = self.tre[i][0]
        
        # return delaunay.find_simplex(x)
    
    def get_U(self,x):
        t = x[0]
        i = int(t*10)
        return self.tre[i][1]
    
    def get_tri(self,x):
        t = x[0]
        i = int(t*10)
        return  self.tre[i][0]
    
    
# def get_vel_snippets(t_span):
#     t_min = t_span[0]
#     t_max = t_span[1]
    
#     velocities= {}
    
#     for i in range(t_min*10,t_max*10):
#         i_span = (i/10,(i+1.5)/10)
        
     
        
#         velocities[i] = get_velocity_data(i_span)
        
#     return velocities


# import scipy.interpolate as spint

def interp_weights(xyz, uvw):
    
    d=3
    
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

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
    
