
import h5py
import pickle
import numpy as np

import ray

from datagenerering import generate_U_txy, generate_ribs, lagra_tre, tre_objekt
import scipy.spatial.qhull as qhull
from scipy.spatial import cKDTree

def lag_tre_multi(t_span, filnamn_inn, filnamn_ut=None):
    
    # a_pool = multiprocessing.Pool()
    
    t_min = t_span[0]
    t_max = t_span[1]
    
    # i = [(i/10,(i+1.5)/10) for i in range(int(t_min)*10,int(t_max)*10)]
    
    # result = a_pool.map(lag_tre, i)
    # jobs = [lag_tre.remote(span, filnamn_inn) for span in i]
    
    ray.init(num_cpus=4)

    with h5py.File(filnamn_inn, 'r') as f:
        U = f.attrs['U']
        L = f.attrs['L']
        rib_width = f.attrs['rib_width']
        
        I = f.attrs['I']
        J = f.attrs['J']
        Umx = np.array(f['Umx'])#[int(t_min*fps):int(t_max*fps),:]
        Vmx = np.array(f['Vmx'])#[int(t_min*fps):int(t_max*fps),:]
        x = np.array(f['x']).reshape(J,I)
        y = np.array(f['y']).reshape(J,I)
        ribs = np.array(f['ribs'])

    u_r = ray.put(Umx)
    v_r = ray.put(Vmx)
    x_r = ray.put(x)
    y_r = ray.put(y)
    ribs_r = ray.put(ribs)

    # experiment = re.search("TONSTAD_([A-Z]*)_", filnamn_inn, re.IGNORECASE).group(1)

    jobs = {lag_tre.remote((i/10,(i+1.5)/10), u_r,v_r,x_r,y_r,I,J,ribs_r, L,rib_width):i for i in range(int(t_min)*10,int(t_max)*10+1)}

    # i_0 =  range(int(t_min)*10,int(t_max)*10)

    trees = {}

    not_ready = list(jobs.keys())
    while True:
        ready, not_ready = ray.wait(not_ready)
        trees[jobs[ready[0]]] = ray.get(ready)[0]

        if len(not_ready)==0:
            break

    kdjob = lag_tre.remote(t_span, u_r,v_r,x_r,y_r,I,J,ribs_r, L, rib_width, nearest=True, kutt= False, inkluder_ribs=True)
    
    kdtre, u, ribs = ray.get(kdjob)
    
    ray.shutdown()
    # trees = dict(zip(i_0, result))
         
    tre_obj = tre_objekt(trees, kdtre, u, ribs)

    if filnamn_ut is None:
        return tre_obj
    else:
        lagra_tre(tre_obj, filnamn_ut)

@ray.remote
def lag_tre(t_span, Umx,Vmx,x,y,I,J,ribs, L, rib_width, nearest=False, kutt=True, inkluder_ribs = False):
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

    U, txy = generate_U_txy(t_span, Umx,Vmx,x,y,I,J,ribs, L, rib_width, kutt)
    
    if (nearest):
        tree = cKDTree(txy)
    else:
        # print(f"Byrjar på delaunay for ({t_min}, {t_max})")
        # start = datetime.datetime.now()
        tree = qhull.Delaunay(txy)
        # print(f"Ferdig med delaunay for ({t_min}, {t_max}, brukte {datetime.datetime.now()-start}")
        # del start
    
    if (inkluder_ribs):
        venstre_ribbe, hogre_ribbe, golv = generate_ribs(ribs, L, rib_width)

        return tree, U, [venstre_ribbe, hogre_ribbe, golv]
    else:
        return tree, U