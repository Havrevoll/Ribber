import matplotlib

from hjelpefunksjonar import f2t
matplotlib.use("Agg")

import numpy as np
import pickle
import matplotlib.pyplot as plt
from f import f
from rib import Rib
from check_collision import check_all_collisions
from matplotlib.patches import Polygon
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['mathtext.fontset'] = 'stix'


partikkelfil = "partikkelsimulasjonar/particles_rib50_Q40_1_scale40_BDF_200_[0.22903447 0.26448726]_40_4e+00_linear.pickle"

with open("partikkelsimulasjonar/particles_rib50_Q40_1_scale40_BDF_200_[0.13389628 0.16396457]_40_4e+00_linear.pickle",'rb') as fil:
    partiklar = pickle.load(fil)

with open("data/rib50_Q40_1_scale40.pickle",'rb') as fil:
    tre = pickle.load(fil)
del fil

ribs = []
for r in tre.ribs:
    ribs.append(Rib(r))

skalering = 40
myDPI = 300

x = (ribs[0].get_rib_middle()[0],ribs[1].get_rib_middle()[0])
x_width = x[1] - x[0]


for p in partiklar:
    if p.index != 75:
        continue
    sti = p.sti_dict
    init = sti['init_time']
    final = sti['final_time']
    plott_array = np.zeros((final+1-init,4))
    for frame in range(init,final+1):
        # drag = drag_component, gravity = gravity_component, added_mass = added_mass_component - 0.5 * rho_self_density * dudt, pressure = pressure_component, lift_component = lift_component, dudt = dudt, dxdt=dxdt
        plott_array[frame-init,:] = np.asarray(sti[frame]['position'])+np.asarray([sti[frame]['loops']*x_width,0,0,0])

    if not np.any(plott_array[:,1]<0):
        continue

    fig, ax  = plt.subplots(figsize=(sti[frame]['loops']*200/myDPI,1000/myDPI),dpi=myDPI)
    ax.plot(plott_array[:400,0], plott_array[:400,1], "ko", markersize=0.5)

    ax.add_patch(Polygon(ribs[0].vertices, facecolor='red'))
    for i in range(9):#sti[frame]['loops']):
        ax.add_patch(Polygon(ribs[1].vertices+ i* np.asarray([x_width, 0]), facecolor='red')) 

    for i in range(10,390,20):
        p.collision = check_all_collisions(p,np.asarray(sti[init+i]['position']),ribs)
        sti[init+i]['forces'] = f(f2t(init+i,skalering),np.asarray(sti[init+i]['position']).reshape(4,1),p,tre,ribs, skalering=skalering,separated=True)
        pilskala = 0.1
        for force,col in {'gravity':'k','drag':'b','added_mass':'r','pressure':'c','lift_component':'m'}.items():
            ax.arrow(plott_array[i,0], plott_array[i,1], float(sti[init+i]['forces'][force][0])*pilskala, float(sti[init+i]['forces'][force][1])*pilskala, width=80, head_width=200, head_length=200, color=col)#, fc=col, ec=None)

    fig.savefig(f"plots/sti_{p.index}.png")
    plt.close()
