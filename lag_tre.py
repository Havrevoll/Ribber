
import datetime
import h5py
# import pickle
import numpy as np

import ray

from datagenerering import generate_U_txy, generate_ribs, lagra_tre, tre_objekt
from scipy.spatial import Delaunay
from scipy.spatial import KDTree

def lag_tre_multi(f_span, filnamn_inn, filnamn_ut=None, skalering=1,linear = True):
    """Lagar eit objekt som kan brukast til interpolering av fartsdata.

    Args:
        f_span (tuple): Ein tuple med 
        filnamn_inn (_type_): _description_
        filnamn_ut (_type_, optional): _description_. Defaults to None.
        skalering (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """  
    
    ray.init(num_cpus=4)#,local_mode=True,include_dashboard=False

    with h5py.File(filnamn_inn, 'r') as f:
        U = f.attrs['U']*skalering**0.5
        L = f.attrs['L']*skalering
        rib_width = f.attrs['rib_width']*skalering
        
        I = f.attrs['I']
        J = f.attrs['J']
        Umx = np.array(f['Umx'])*skalering**0.5
        Vmx = np.array(f['Vmx'])*skalering**0.5
        x = np.array(f['x']).reshape(J,I)*skalering
        y = np.array(f['y']).reshape(J,I)*skalering
        ribs = np.array(f['ribs'])*skalering

    u_r = ray.put(Umx)
    v_r = ray.put(Vmx)
    x_r = ray.put(x)
    y_r = ray.put(y)
    ribs_r = ray.put(ribs)
    trees = {}
    
    if linear:
        jobs = {lag_tre.remote((i,i+2), u_r,v_r,x_r,y_r,I,J,ribs_r, L,rib_width):i for i in range(f_span[0],f_span[1])}


        not_ready = list(jobs.keys())
        while True:
            ready, not_ready = ray.wait(not_ready)
            trees[jobs[ready[0]]] = ray.get(ready)[0]

            if len(not_ready)==0:
                break

    kdjob = lag_tre.remote(f_span, u_r,v_r,x_r,y_r,I,J,ribs_r, L, rib_width, nearest=True, kutt= False, inkluder_ribs=True)
    
    kdtre, u, ribs = ray.get(kdjob)
    
    ray.shutdown()
    # trees = dict(zip(i_0, result))
         
    tre_obj = tre_objekt(trees, kdtre, u, ribs)

    if filnamn_ut is None:
        return tre_obj
    else:
        lagra_tre(tre_obj, filnamn_ut)

@ray.remote
def lag_tre(f_span, Umx,Vmx,x,y,I,J,ribs, L, rib_width, nearest=False, kutt=True, inkluder_ribs = False):
    """Lagar eit delaunay- eller kd-tre ut frå f_span og ei hdf5-fil.

    Args:
        f_span (tuple): Tid frå og til
        filnamn (string): Filnamn på hdf5-fila
        nearest (bool, optional): lineær om false, kdtree om true?  Defaults to False.
        kutt (bool, optional): Kutt av data til eit lite område?. Defaults to True.
        inkluder_ribs (bool, optional): Ta med ribbedata. Defaults to False.
        kutt_kor (list, optional): Koordinatane til firkanten som skal kuttast. Defaults to [-35.81, 64.19 , -25, 5].

    Returns:
         tuple: Delaunay eller kd-tre, U pluss ev. ribber
    """

    U, txy = generate_U_txy(f_span, Umx,Vmx,x,y,I,J,ribs, L, rib_width, kutt)
    
    if (nearest):
        tree = KDTree(txy)
    else:
        print(f"Byrjar på delaunay for ({f_span})")
        start = datetime.datetime.now()
        tree = Delaunay(txy)
        print(f"Ferdig med delaunay for ({f_span}, brukte {datetime.datetime.now()-start}")
        del start
    
    if (inkluder_ribs):
        venstre_ribbe, hogre_ribbe, golv = generate_ribs(ribs, L, rib_width)

        return tree, U, [venstre_ribbe, hogre_ribbe, golv]
    else:
        return tree, U


if __name__ == "__main__":
    # Dette er det som skjer i "lag_tre_multi":

    filnamn_inn = "data/rib50_Q40_1.hdf5"
    skalering = 40
    f_span = (867,869)
    kutt = True
    with h5py.File(filnamn_inn, 'r') as f:
        U = f.attrs['U']*skalering**0.5
        L = f.attrs['L']*skalering
        rib_width = f.attrs['rib_width']*skalering
        
        I = f.attrs['I']
        J = f.attrs['J']
        Umx = np.array(f['Umx'])*skalering**0.5
        Vmx = np.array(f['Vmx'])*skalering*0.5
        x = np.array(f['x']).reshape(J,I)*skalering
        y = np.array(f['y']).reshape(J,I)*skalering
        ribs = np.array(f['ribs'])*skalering

    # Herifrå er det det som skjer i "lag_tre":
    U, txy = generate_U_txy(f_span, Umx,Vmx,x,y,I,J,ribs, L, rib_width, kutt)

    tree = Delaunay(txy)
    
    venstre_ribbe, hogre_ribbe, golv = generate_ribs(ribs, L, rib_width)
    
#     tre = lag_tre((0,2), Umx,Vmx,x,y,I,J,ribs, L, rib_width)