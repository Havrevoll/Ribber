# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 12:37:35 2021

@author: havrevol
"""
import h5py
import pickle
import numpy as np

# from hjelpefunksjonar import ranges, finn_fil
import datetime
import scipy.spatial.qhull as qhull
from scipy.spatial import cKDTree
import re

# import multiprocessing

import ray


# filnamn = "../two_q40.hdf5" #finn_fil(["D:/Tonstad/utvalde/Q40.hdf5", "C:/Users/havrevol/Q40.hdf5", "D:/Tonstad/Q40.hdf5"])

# pickle_fil = finn_fil(["../Q40_60s.pickle", "D:/Tonstad/Q40_20s.pickle", "C:/Users/havrevol/Q40_20s.pickle", "D:/Tonstad/Q40_2s.pickle"])

# print("pickle fil er ", pickle_fil) 

def lag_tre_multi(t_span, filnamn_inn, filnamn_ut=None):
    
    # a_pool = multiprocessing.Pool()
    
    t_min = t_span[0]
    t_max = t_span[1]
    
    # i = [(i/10,(i+1.5)/10) for i in range(int(t_min)*10,int(t_max)*10)]
    
    # result = a_pool.map(lag_tre, i)
    # jobs = [lag_tre.remote(span, filnamn_inn) for span in i]
    
    ray.init(num_cpus=4)

    with h5py.File(filnamn_inn, 'r') as f:
        Umx = np.array(f['Umx'])#[int(t_min*fps):int(t_max*fps),:]
        Vmx = np.array(f['Vmx'])#[int(t_min*fps):int(t_max*fps),:]
        (I,J) = (int(np.array(f['I'])),int(np.array(f['J'])))
        x = np.array(f['x']).reshape(J,I)
        y = np.array(f['y']).reshape(J,I)
        ribs = np.array(f['ribs'])

    u_r = ray.put(Umx)
    v_r = ray.put(Vmx)
    x_r = ray.put(x)
    y_r = ray.put(y)
    ribs_r = ray.put(ribs)

    experiment = re.search("TONSTAD_([A-Z]*)_", filnamn_inn, re.IGNORECASE).group(1)

    jobs = {lag_tre.remote((i/10,(i+1.5)/10), u_r,v_r,x_r,y_r,I,J,ribs_r, experiment):i for i in range(int(t_min)*10,int(t_max)*10)}

    # i_0 =  range(int(t_min)*10,int(t_max)*10)

    trees = {}

    not_ready = list(jobs.keys())
    while True:
        ready, not_ready = ray.wait(not_ready)
        trees[jobs[ready[0]]] = ray.get(ready)[0]

        if len(not_ready)==0:
            break

    ray.shutdown()
    # trees = dict(zip(i_0, result))
         
    if filnamn_ut is None:
        return trees
    else:
        lagra_tre(trees, filnamn_ut)

@ray.remote
def lag_tre(t_span, Umx,Vmx,x,y,I,J,ribs, experiment, nearest=False, kutt=True, inkluder_ribs = False, kutt_kor = [-35.81,64.19 , -25, 5]):
    """Lagar eit delaunay- eller kd-tre ut frå t_span og ei hdf5-fil.

    Args:
        t_span (tuple): Tid frå og til
        filnamn (string): Filnamn på hdf5-fila
        nearest (bool, optional): lineær?  Defaults to False.
        kutt (bool, optional): Kutt av data til eit lite område?. Defaults to True.
        inkluder_ribs (bool, optional): Ta med ribbedata. Defaults to False.
        kutt_kor (list, optional): Koordinatane til firkanten som skal kuttast. Defaults to [-35.81, 64.19 , -25, 5].

    Returns:
         tuple: Delaunay eller kd-tre, U pluss ev. ribber
    """

    t_min = t_span[0]
    t_max = t_span[1]
    fps = 20
    Umx = Umx[int(t_min*fps):int(t_max*fps),:]
    Vmx = Vmx[int(t_min*fps):int(t_max*fps),:]

    # piv_range = ranges(kutt)
    
    x = np.copy(x)
    y = np.copy(y)
    
    # finn x- og y-koordinatane for kuttinga

    Umx_reshape = np.copy(Umx.reshape(len(Umx), J, I))
    Vmx_reshape = np.copy(Vmx.reshape(len(Vmx), J, I))
    # Umx_reshape = np.zeros((len(Umx), J+1,I))
    # Vmx_reshape = np.zeros((len(Vmx), J+1,I))
    # x = np.vstack((x,x[0,:]))
    # y = np.vstack((y,np.full((1,I), -98.81)))
    
    # Umx_reshape[:,:J,:] = Umx.reshape((len(Umx),J,I))[:,piv_range[0],piv_range[1]]
    # Vmx_reshape[:,:J,:] = Vmx.reshape((len(Vmx),J,I))[:,piv_range[0],piv_range[1]]
    
    # axis0 = np.take_along_axis(ribs[:,:,0], np.argpartition(ribs[:,:,1],-2),1)[:,-2:].T
    # axis1= np.take_along_axis(ribs[:,:,1], np.argpartition(ribs[:,:,1],-2),1)[:,-2:].T
    # for rib in np.stack((axis0,axis1),axis=0).swapaxes(0,2):
        


    if (experiment == "TWO"):
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

        if (experiment == "THREE"):
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
        kutt_kor = [ribs[v_r+1,0]-25, ribs[v_r+1,0]+(ribs[h_r,0] - ribs[v_r+1,0])+25, ribs[v_r+1,1]-24, ribs[v_r+1,1]+6] # [-35.81,64.19 , -25, 5]
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
        
    t_3d,y_3d,x_3d = np.meshgrid(np.arange(t_min*fps,t_max*fps)/fps, y[:,0], x[0,:], indexing='ij')
    y_3d[...] = y
    x_3d[...] = x
    t_lang = t_3d.ravel()[nonan]
    x_lang = x_3d.ravel()[nonan]
    y_lang = y_3d.ravel()[nonan]
        
    txy = np.vstack((t_lang,x_lang,y_lang)).T
    
    if (nearest):
        tree = cKDTree(txy)
    else:
        # print(f"Byrjar på delaunay for ({t_min}, {t_max})")
        # start = datetime.datetime.now()
        tree = qhull.Delaunay(txy)
        # print(f"Ferdig med delaunay for ({t_min}, {t_max}, brukte {datetime.datetime.now()-start}")
        # del start
    
    if (inkluder_ribs):
        #venstre ribbe,
        venstre_ribbe = np.zeros((4,2))
        
        venstre_ribbe[0] = ribs[v_r+1]
        venstre_ribbe[1] = ribs[v_r+2]
        venstre_ribbe[3] = [ribs[v_r+1,0]-50, ribs[v_r+1,1] + (-50) * (ribs[v_r,1] - ribs[v_r+1,1])/(ribs[v_r,0] - ribs[v_r+1,0])]
        venstre_ribbe[2] = venstre_ribbe[1] + venstre_ribbe[3] - venstre_ribbe[0]

        hogre_ribbe = np.zeros((4,2))
        hogre_ribbe[0] = ribs[h_r-1]
        hogre_ribbe[1] = ribs[h_r]
        hogre_ribbe[2] = [ribs[h_r,0]+50, ribs[h_r,1] + 50 * (ribs[h_r+1,1] - ribs[h_r,1])/(ribs[h_r + 1,0] - ribs[h_r,0])]
        hogre_ribbe[3] = hogre_ribbe[0] + hogre_ribbe[2] - hogre_ribbe[1]

        golv = np.zeros((4,2))
        golv[0] = ribs[golv_nr]
        golv[1] = ribs[golv_nr+1]
        golv[2] = ribs[golv_nr+1] + np.array([0,-20])
        golv[3] = ribs[golv_nr] + np.array([0,-20])
        
        

        return tree, U, [venstre_ribbe, hogre_ribbe, golv]
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

# def hent_tre(fil=pickle_fil):
#     with open(fil, 'rb') as f:
#         tri = pickle.load(f)
 
#     return tri

class tre_objekt:
    def __init__(self, pickle_fil, hdf5_fil, t_span):
        with open(pickle_fil, 'rb') as f:
            self.tre = pickle.load(f)
        for t in self.tre:
            if type(self.tre[t]) is list:
                self.tre[t] = self.tre[t][0]

        
        with h5py.File(hdf5_fil, 'r') as f:
            Umx = np.array(f['Umx'])#[int(t_min*fps):int(t_max*fps),:]
            Vmx = np.array(f['Vmx'])#[int(t_min*fps):int(t_max*fps),:]
            (I,J) = (int(np.array(f['I'])),int(np.array(f['J'])))
            x = np.array(f['x']).reshape(J,I)
            y = np.array(f['y']).reshape(J,I)
            ribs = np.array(f['ribs'])

        ray.init()

        u_r = ray.put(Umx)
        v_r = ray.put(Vmx)
        x_r = ray.put(x)
        y_r = ray.put(y)
        ribs_r = ray.put(ribs)

        experiment = re.search("TONSTAD_([A-Z]*)_", str(pickle_fil), re.IGNORECASE).group(1)
       
        job = lag_tre.remote(t_span, u_r,v_r,x_r,y_r,I,J,ribs_r, experiment, nearest=True, kutt= False, inkluder_ribs=True)
        self.kdtre, self.U_kd, self.ribs = ray.get(job)
        ray.shutdown()
    
    
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
    
