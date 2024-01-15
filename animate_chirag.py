import datetime
import pickle
import numpy as np
import matplotlib
from pathlib import Path
import subprocess
# import ray
from kornfordeling import PSD_plot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from cycler import cycler
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'
from matplotlib import animation
from scipy.interpolate import interp1d

import random
import h5py
from hjelpefunksjonar import ranges, draw_rect
from math import hypot
from rib import Rib

def sti_animasjon(ribs, t_span, hdf5_fil, utfilnamn=Path("stiQ40_flow.mp4"),  fps=60, slow = 1):
    
    t_min = t_span[0]
    t_max = t_span[1]

    steps = (t_max-t_min) * fps
    
    # t_list = np.arange(t_min*fps,t_max*fps)/fps
    t_list = np.arange(t_min*fps,t_max*fps)/fps
    t_list_part = np.arange(t_min*20,t_max*20)/20
    
    # piv_range = ranges()

    with h5py.File(hdf5_fil, 'r') as dataset:
        (I,J)=dataset.attrs['I'],dataset.attrs['J']
    
        Umx = np.array(dataset['Umx'])[t_min*20:t_max*20,:]
        Umx_reshape = Umx.reshape((len(Umx),J,I))#[:,piv_range[0],piv_range[1]]
        Vmx = np.array(dataset['Vmx'])[t_min*20:t_max*20,:]
        Vmx_reshape = Vmx.reshape((len(Vmx),J,I))#[:,piv_range[0],piv_range[1]]

        # ribs = np.array(dataset['ribs'])
        
        x = np.array(dataset['x'])
        y = np.array(dataset['y'])

    x_reshape = x.reshape(J,I)#[piv_range]
    y_reshape = y.reshape(J,I)#[piv_range]
            
    V_mag_reshape = np.hypot(Umx_reshape, Vmx_reshape)        
    # V_mag_reshape = np.hypot(U[2], U[3])
    V_mag_interp = interp1d(range(t_min*fps,t_max*fps,int(fps/20)), V_mag_reshape, axis=0)

    myDPI = 300

    fig, ax = plt.subplots(figsize=(1234/myDPI,1080/myDPI),dpi=myDPI)
    
    field = ax.imshow(V_mag_reshape[0,:,:], extent=[x_reshape[0,0],x_reshape[0,-1], y_reshape[-1,0], y_reshape[0,0]], interpolation='none')
        
    ax.set_xlim([x_reshape[0,0],x_reshape[0,-1]])
    draw_rect(ax, ribs)

    filmstart = [datetime.datetime.now()]

    def nypkt(i):
        field.set_data(V_mag_interp(i))
        return 
    
    #ax.axis('equal')
    # ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,steps),interval=50)
    FFwriter = animation.FFMpegWriter(fps=fps)
    ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(0,steps-int(fps/20)),interval=(slow*1000/(fps)), blit=False)
    # plt.show()
    
    starttid = datetime.datetime.now()
    ani.save(utfilnamn, writer=FFwriter)
    print(f"Brukte {datetime.datetime.now()-starttid} på å lagra filmen")
    plt.close()



if __name__ == "__main__":


    with open("data/rib50_Q40_1.pickle",'rb') as f: 
        ribs = [Rib(rib) for rib in pickle.load(f).ribs]

    sti_animasjon(ribs=ribs,t_span=(0,120),hdf5_fil=Path("data/rib50_Q40_1.hdf5"),fps=60, slow=1)
