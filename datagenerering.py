# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:37:35 2021

@author: havrevol
"""
import h5py
import pickle
import numpy as np

# from hjelpefunksjonar import ranges, finn_fil
# import datetime
# import re

# filnamn = "../two_q40.hdf5" #finn_fil(["D:/Tonstad/utvalde/Q40.hdf5", "C:/Users/havrevol/Q40.hdf5", "D:/Tonstad/Q40.hdf5"])

# pickle_fil = finn_fil(["../Q40_60s.pickle", "D:/Tonstad/Q40_20s.pickle", "C:/Users/havrevol/Q40_20s.pickle", "D:/Tonstad/Q40_2s.pickle"])

# print("pickle fil er ", pickle_fil) 


def generate_ribs(ribs, L, rib_width):
    v_r, golv_nr, h_r, _, _, _, _, _, _, _, _, _ = get_essential_coordinates(L,rib_width)

    venstre_ribbe = np.zeros((4,2))
    
    venstre_ribbe[0] = ribs[v_r + 1]
    venstre_ribbe[1] = ribs[v_r + 2]
    venstre_ribbe[3] = [ribs[v_r + 1, 0] - rib_width, ribs[v_r + 1, 1] + (-rib_width) * (ribs[v_r,1] - ribs[v_r+1,1])/(ribs[v_r,0] - ribs[v_r+1,0])]
    venstre_ribbe[2] = venstre_ribbe[1] + venstre_ribbe[3] - venstre_ribbe[0]

    hogre_ribbe = np.zeros((4,2))
    hogre_ribbe[0] = ribs[h_r-1]
    hogre_ribbe[1] = ribs[h_r]
    hogre_ribbe[2] = [ribs[h_r,0] + rib_width, ribs[h_r,1] + rib_width * (ribs[h_r+1,1] - ribs[h_r,1])/(ribs[h_r + 1,0] - ribs[h_r,0])]

    hogre_ribbe[3] = hogre_ribbe[0] + hogre_ribbe[2] - hogre_ribbe[1]

    golv = np.zeros((4,2))
    golv[0] = ribs[golv_nr]
    golv[1] = ribs[golv_nr+1]
    golv[2] = ribs[golv_nr+1] + np.array([0,-rib_width/2])
    golv[3] = ribs[golv_nr] + np.array([0,-rib_width/2])
    
    return venstre_ribbe, hogre_ribbe, golv

def generate_U_txy(f_span, Umx,Vmx,x,y,I,J,ribs, L, rib_width, kutt=True):
    f_min = f_span[0]
    f_max = f_span[1]
    # fps = 20
    Umx = Umx[f_min:f_max,:]
    Vmx = Vmx[f_min:f_max,:]

    x = np.copy(x)
    y = np.copy(y)
    
    # finn x- og y-koordinatane for kuttinga

    Umx_reshape = np.copy(Umx.reshape(len(Umx), J, I))
    Vmx_reshape = np.copy(Vmx.reshape(len(Vmx), J, I))

    v_r, golv_nr, h_r, v_r_rad, v_r_kol, v_r_tjukk, golv_rad1, golv_rad2, golv_skifte, h_r_rad, h_r_kol, h_r_tjukk = get_essential_coordinates(L,rib_width)

    # Venstre ribbe
    x0 = ribs[v_r,0]  #-60.79
    x1 = ribs[v_r+1,0]  #-10.84
    y0 = ribs[v_r,1]  #-.8265
    y1 = ribs[v_r+1,1]  #-1.1020
    
    # y[64,18:55] = -.8265+ (x[0,18:55] + 60.79)* (-1.1020+0.8265)/(-10.84+60.79)
    y[v_r_rad,0:v_r_kol] = y0 + (x[0,0:v_r_kol] - x0)* (y1 - y0)/(x1 - x0)
    
    # y[63,0:54]=-1.01
    Umx_reshape[:,v_r_rad:v_r_rad+v_r_tjukk,0:v_r_kol]=0
    Vmx_reshape[:,v_r_rad:v_r_rad+v_r_tjukk,0:v_r_kol]=0
    
    x0 = ribs[h_r,0]   #39.028
    x1 = ribs[h_r+1,0]   #89.075
    y0 = ribs[h_r,1]   #0.0918
    y1 = ribs[h_r+1,1]   #0.0918
    
    x[h_r_rad:h_r_rad+h_r_tjukk,h_r_kol] = x0
    y[h_r_rad,h_r_kol:] = y0  + (x[0,h_r_kol:] - x0)* (y1 - y0)/(x1 - x0)
    Umx_reshape[:,h_r_rad:h_r_rad+h_r_tjukk,h_r_kol:]=0
    Vmx_reshape[:,h_r_rad:h_r_rad+h_r_tjukk,h_r_kol:]=0
    
    
    x0 = ribs[golv_nr,0]  #-93.2075
    x1 = ribs[golv_nr+1,0]  #93.3
    y0 = ribs[golv_nr,1]  #-72.6375
    y1 = ribs[golv_nr+1,1]  #-74.8415
    
    y [golv_rad1,0:golv_skifte] = y0  + (x[0,0:golv_skifte] - x0)* (y1 - y0)/(x1 - x0)
    y [golv_rad2,golv_skifte:] = y0  + (x[0,golv_skifte:] - x0)* (y1 - y0)/(x1 - x0)
    Umx_reshape[:,golv_rad1,0:golv_skifte] = 0
    Vmx_reshape[:,golv_rad1,0:golv_skifte] = 0
    Umx_reshape[:,golv_rad2,golv_skifte:] = 0
    Vmx_reshape[:,golv_rad2,golv_skifte:] = 0

    
    
    if kutt:
        kutt_kor = [ribs[v_r+1,0]-rib_width/2, ribs[v_r+1,0]+(ribs[h_r,0] - ribs[v_r+1,0]) + rib_width/2, ribs[v_r+1,1]-rib_width/2, ribs[v_r+1,1] + 10 * rib_width * 0.02] # [-35.81,64.19 , -25, 5]
        x1 = x[0,:]
        y1=y[:,0]

        Umx_reshape = Umx_reshape[:,:,(x1 >kutt_kor[0]) & (x1 < kutt_kor[1] )][:,((y1 > kutt_kor[2]) & (y1 < kutt_kor[3])),:]
        Vmx_reshape = Vmx_reshape[:,:,(x1 >kutt_kor[0]) & (x1 < kutt_kor[1] )][:,((y1 > kutt_kor[2]) & (y1 < kutt_kor[3])),:]
        
        x = x[:, (x1 >kutt_kor[0]) & (x1 < kutt_kor[1] )][((y1 > kutt_kor[2]) & (y1 < kutt_kor[3])), :]
        y = y[:, (x1 >kutt_kor[0]) & (x1 < kutt_kor[1] )][((y1 > kutt_kor[2]) & (y1 < kutt_kor[3])), :]

    Umx_lang = Umx_reshape.ravel()
    Vmx_lang = Vmx_reshape.ravel()
        
    nonan = np.invert(np.isnan(Umx_lang))
    
    U = np.array([Umx_lang[nonan], Vmx_lang[nonan]])
        
    t_3d,y_3d,x_3d = np.meshgrid(np.arange(f_min,f_max), y[:,0], x[0,:], indexing='ij')
    y_3d[...] = y
    x_3d[...] = x
    t_lang = t_3d.ravel()[nonan]
    x_lang = x_3d.ravel()[nonan]
    y_lang = y_3d.ravel()[nonan]
        
    txy = np.vstack((t_lang,x_lang,y_lang)).T

    return U, txy

def get_essential_coordinates(L,rib_width):
    skala = round(rib_width/50)
    if (L == 50*skala):
        v_r = 1
        golv_nr = 8
        h_r = 16

        v_r_rad = 64
        v_r_kol = 56
        v_r_tjukk = 3

        golv_rad1 = 113
        golv_rad2 = 114
        golv_skifte = 55

        h_r_rad = 63
        h_r_kol = 88
        h_r_tjukk = 6
        
    else:
        v_r = 1
        golv_nr = 6
        h_r = 11

        if (L == 75*skala):
            v_r_rad = 70
            v_r_kol = 29
            v_r_tjukk = 6

            golv_rad1 = 122
            golv_rad2 = 122
            golv_skifte = 50

            h_r_rad = 69
            h_r_kol = 80
            h_r_tjukk = 6
        else:
            v_r_rad = 65
            v_r_kol = 61
            v_r_tjukk = 6

            golv_rad1 = 123
            golv_rad2 = 122
            golv_skifte = 46

            h_r_rad = 65
            h_r_kol = 80
            h_r_tjukk = 6

    return v_r, golv_nr, h_r, v_r_rad, v_r_kol, v_r_tjukk, golv_rad1, golv_rad2, golv_skifte, h_r_rad, h_r_kol, h_r_tjukk

def lagra_tre(tre, fil):
    with open(fil, 'wb') as f:
        pickle.dump(tre, f)

# def hent_tre(fil=pickle_fil):
#     with open(fil, 'rb') as f:
#         tri = pickle.load(f)
 
#     return tri

class tre_objekt:
    def __init__(self, delaunay, kdtre, U_kd, ribs):
        """Eit objekt med relevant informasjon

        Args:
            delaunay (dict):  delaunay er ein dict med mange tuple returnert frå lag_tre: (tree, U)
            kdtre (_type_): _description_
            U_kd (_type_): _description_
            ribs (_type_): _description_
        """
        self.tre = delaunay
        for t in self.tre:
            if type(self.tre[t]) is list:
                self.tre[t] = self.tre[t][0]
        self.kdtre = kdtre
        self.U_kd = U_kd
        self.ribs = ribs
    
    
    # def find_simplex(self, x):
    #     t = x[0]
    #     i = int(t*10)
        
        # delaunay = self.tre[i][0]
        
        # return delaunay.find_simplex(x)
    
    def get_U(self,x):
        # t = x[0]
        # i = int(t*10)

        return self.tre[x[0]][1]
    
    def get_tri(self,x):
        # t = x[0]
        # i = int(t*10)
        # return  self.tre[i][0]
        return  self.tre[x[0]][0]
    
    def get_tri_og_U(self,t):
        return self.tre[int(t)]
    
    def get_kd_U(self, tx):
        while True:
                try:
                    self.U_kd[:,self.kdtre.query(np.swapaxes(tx, -2,-1))[1]]
                    break
                except IndexError:
                    tx[np.abs(tx)>1e100] /= 1e10
                    
                
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
    
