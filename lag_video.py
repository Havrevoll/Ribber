import datetime
import pickle
import numpy as np
import matplotlib
from pathlib import Path

from kornfordeling import PSD_plot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from cycler import cycler
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'
from matplotlib import animation

import random
import h5py
from hjelpefunksjonar import ranges, draw_rect
from math import hypot

# filnamn = "../Q40.hdf5" #finn_fil(["D:/Tonstad/utvalde/Q40.hdf5", "C:/Users/havrevol/Q40.hdf5", "D:/Tonstad/Q40.hdf5"])

def lag_video(partikkelfil, filmfil, hdf5_fil, ribs, t_span, fps=20):
    with open(partikkelfil, 'rb') as f:
        particle_list = pickle.load(f)

    start = datetime.datetime.now()
    sti_animasjon(particle_list, ribs, t_span, hdf5_fil=hdf5_fil, utfilnamn=filmfil, fps=fps)
    print(f"Brukte {datetime.datetime.now() - start} på å lagra filmen")

def sti_animasjon(partiklar, ribs, t_span, hdf5_fil, utfilnamn=Path("stiQ40.mp4"),  fps=20 ):
    
    # piv_range = ranges()
    
    # with h5py.File(filnamn, mode='r') as f:
    #     x, y = np.array(f['x']), np.array(f['y'])
        
    #     I, J = int(np.array(f['I'])),int(np.array(f['J']))
               
    #     x_reshape = x.reshape((127,126))[piv_range]
    #     y_reshape = y.reshape((127,126))[piv_range]

    t_min = t_span[0]
    t_max = min(t_span[1], int(max([p.sti_dict['final_time'] for p in partiklar]))+5)
    
    steps = (t_max-t_min) * fps
    
    t_list = np.arange(t_min*fps,t_max*fps)/fps
    
    # piv_range = ranges()

    with h5py.File(hdf5_fil, 'r') as dataset:
        (I,J)=(int(np.array(dataset['I'])),int(np.array(dataset['J'])))
    
        Umx = np.array(dataset['Umx'])[t_min*fps:t_max*fps,:]
        Umx_reshape = Umx.reshape((len(Umx),J,I))#[:,piv_range[0],piv_range[1]]
        Vmx = np.array(dataset['Vmx'])[t_min*fps:t_max*fps,:]
        Vmx_reshape = Vmx.reshape((len(Vmx),J,I))#[:,piv_range[0],piv_range[1]]

        # ribs = np.array(dataset['ribs'])
        
        x = np.array(dataset['x'])
        y = np.array(dataset['y'])

    x_reshape = x.reshape(J,I)#[piv_range]
    y_reshape = y.reshape(J,I)#[piv_range]
            
    V_mag_reshape = np.hypot(Umx_reshape, Vmx_reshape)        
    # V_mag_reshape = np.hypot(U[2], U[3])
       
    myDPI = 300
    fig = plt.figure(dpi=myDPI, figsize=(7.2,4.8))
    gs = GridSpec(2, 2, width_ratios=[2, 1] )
    ax = fig.add_subplot( gs[:, :-1] )
    ax2 = fig.add_subplot( gs[0, -1])
    ax3 = fig.add_subplot( gs[-1, -1])

    # fig, ax = plt.subplots(figsize=(800/myDPI,700/myDPI),dpi=myDPI)
    
    field = ax.imshow(V_mag_reshape[0,:,:], extent=[x_reshape[0,0],x_reshape[0,-1], y_reshape[-1,0], y_reshape[0,0]], interpolation='none')
    # particle, =ax.plot(sti[:,0,1], sti[:,0,2], color='black', marker='o', linestyle=' ', markersize=1)
    
    # https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
    # particle = 
    
    # sti = np.array([part.sti for part in partiklar])    
    
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    
    # F8B195   F67280   C06C84   6C5B7B   355C7D 
    # A7226E   EC2049   F26B38   F7DB4F   2F9599
    # A8A7A7   CC527A   E8175D   474747   363636 
    # E5FCC2   9DE0AD   45ADA8   547980   594F4F
    # Palettar kan finnast her: https://digitalsynopsis.com/design/minimal-web-color-palettes-combination-hex-code/
    # https://digitalsynopsis.com/design/how-to-make-color-palettes-from-one-color/

    #Denne oppskrifta lagar eit nytt fargesett: 
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/markevery_prop_cycle.html#sphx-glr-gallery-lines-bars-and-markers-markevery-prop-cycle-py
    
    
#    colors = ['#1f77b4',          '#ff7f0e',          '#2ca02c',          '#d62728',          '#9467bd',          '#8c564b',          '#e377c2',          '#7f7f7f',          '#bcbd22',          '#17becf',          '#1a55FF']
    
    # colors = [ '#F8B195', '#F67280', '#C06C84', '#6C5B7B', '#355C7D', '#A7226E', '#EC2049', '#F26B38', '#F7DB4F', '#2F9599', '#A8A7A7', '#CC527A', '#E8175D', '#474747', '#363636', '#E5FCC2', '#9DE0AD', '#44AA57', '#547980', '#594F4F']
    # matplotlib.rcParams['axes.prop_cycle'] = cycler(color=colors)


    colors = [f"#{random.randint(0,16**6-1):0{6}X}" for _ in range(len(partiklar))]
    caught = []
    uncaught = []

    for part,color in zip(partiklar, colors):
        circle = plt.Circle((-100, -100), part.radius*5, color=color, visible = False)
        ax.add_patch(circle)
        part.circle = circle
        part.annotation  = ax.annotate("{:.2f} mm".format(part.diameter), xy=(-100,-100),#, visible = False, #np.interp(0,part.sti[:,0],part.sti[:,1]), np.interp(0,part.sti[:,0],part.sti[:,2])), 
                        xycoords="data", xytext=(random.random()*8,random.random()*8), fontsize=5, textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=0", color=color))
        # print("{} {} {} {}".format(part.atol,part.rtol,part.method, color))
        # gamal xytext: random.uniform(-20,20), random.uniform(-20,20)
        if part.sti_dict[round(part.sti_dict['init_time']*100)]['caught']:
            caught.append(part.diameter)
        else:
            uncaught.append(part.diameter)
        
    ax.set_xlim([x_reshape[0,0],x_reshape[0,-1]])
    draw_rect(ax, ribs)

    bins = 2.0**(np.arange(-6,3))
    x1 = PSD_plot(np.asarray(uncaught), bins)
    uncaught_plot, = ax2.semilogx(bins, x1, color="blue", label="kornfordeling")
    ax2.set_xticks(bins[0::2])
    ax2.set_xticklabels(bins[0::2])
    ax2.set_xlabel("d [mm]")
    ax2.set_ylabel("% masse partiklar passert")
    ax2.set_ylim(0,1)

    x2 = PSD_plot(np.asarray(caught), bins)
    caught_plot, = ax3.semilogx(bins, x2, color="blue", label="kornfordeling")
    ax3.set_xticks(bins[0::2])
    ax3.set_xticklabels(bins[0::2])
    ax3.set_xlabel("d [mm]")
    ax3.set_ylabel("% masse partiklar passert")
    ax3.set_ylim(0,1)

    uncaught_text = fig.text(0.15,0.6, f"Free: {len(uncaught)}")
    caught_text = fig.text(0.15,0.4, f"Trapped: {len(uncaught)}")
    filmstart = [datetime.datetime.now()]

    def nypkt(i):
        field.set_data(V_mag_reshape[i,:,:])
        # particle.set_data(sti[:,i,1], sti[:,i,2])
        t = t_list[i]

        caught = []
        uncaught = []
        caught_mass = 0
        uncaught_mass = 0
        
        # https://stackoverflow.com/questions/16527930/matplotlib-update-position-of-patches-or-set-xy-for-circles
        for part in partiklar:
            
            if part.sti_dict[max(round(part.sti_dict['init_time']*100), min(round(t*100), round(part.sti_dict['final_time']*100))) ]['caught']:
                caught.append(part.diameter)
                caught_mass += part.mass
            else:
                uncaught.append(part.diameter)
                uncaught_mass += part.mass

            if t>= part.sti_dict['init_time'] and t <= part.sti_dict['final_time']:
                part.circle.center = part.sti_dict[round(t*100)]['position'][0:2]
                # part.circle.center = part.sti[i,1], part.sti[i,2]
                part.annotation.xy = part.circle.center
                # part.annotation.xy = (np.interp(t,part.sti[:,0],part.sti[:,1], left=-100), np.interp(t,part.sti[:,0],part.sti[:,2], left=-100) )
                
                if not part.circle.get_visible():
                    part.circle.set_visible(True)
                    part.annotation.set_visible(True)

            else:
                part.circle.set_visible(False)
                part.annotation.set_visible(False)

        uncaught_plot.set_ydata(PSD_plot(np.asarray(uncaught), bins))
        caught_plot.set_ydata(PSD_plot(np.asarray(caught), bins))
        uncaught_text.set_text(f"Free: {len(uncaught)}\nMass: {1e6*uncaught_mass:.2f} mg")
        caught_text.set_text(f"Trapped: {len(caught)}\nMass: {1e6*caught_mass:.2f} mg")

        # if t % 100 == 0:
        #     print(f"Har laga {t:>6.2f} av filmen, brukar {datetime.datetime.now()- filmstart[0]} på kvart bilete")
        # filmstart[0] = datetime.datetime.now()

        circles = [p.circle for p in partiklar]
        annotations = [p.annotation for p in partiklar]

        return circles + annotations
    
    #ax.axis('equal')
    # ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(1,steps),interval=50)
    ani = animation.FuncAnimation(fig, nypkt, frames=np.arange(0,steps),interval=int(1000/fps), blit=True)
    # plt.show()
    
    starttid = datetime.datetime.now()
    ani.save(utfilnamn)
    print(f"Brukte {datetime.datetime.now()-starttid} på å lagra filmen")
    plt.close()