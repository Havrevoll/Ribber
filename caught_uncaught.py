import re
import matplotlib
from IPython import get_ipython
from constants import ρ_p
from math import pi as π, sqrt
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import ray

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def d2m(d):
    return d**3 * π/6 * ρ_p

if not isnotebook():
    import matplotlib
    matplotlib.use("Agg")
else:
    # %matplotlib widget
    get_ipython().magic('matplotlib widget')


@ray.remote
def oppsummer(fil):
    samling = []

    if True:
        print(fil.name)
        fil_L = re.search("rib(\d\d)", fil.name).group(1)
        fil_Q = re.search("Q(\d?\d\d)", fil.name).group(1)
        fil_nummer = re.search("(\d)_BDF", fil.name).group(1)
        fil_sim_nummer = re.search("_(\d?)\.pickle", fil.name)
        if fil_sim_nummer:
            fil_sim_nummer = fil_sim_nummer.group(1)
        else:
            fil_sim_nummer = 1

        assert fil.exists()

        caught =0
        uncaught = 0
        with open(fil,'rb') as fila:
            pa = pickle.load(fila)
        del fila

        # samling.append(

        for p in pa:
            try:
                caught = p.sti_dict[int(round(p.sti_dict['final_time']*100))]['caught']
                velocity = sqrt(p.sti_dict[int(round(p.sti_dict['final_time']*100))]['position'][2]**2 + p.sti_dict[int(round(p.sti_dict['final_time']*100))]['position'][3]**2 )
                if not caught and velocity < 0.001:
                    caught = True
                samling.append(dict(L = fil_L, Q = fil_Q, forsøk=fil_nummer, simulering = fil_sim_nummer,d=p.diameter,caught=caught))
            except Exception as e:
                raise Exception(f"partikkel nr {p.index}").with_traceback(e.__traceback__)
    return samling


alle = {25:{20:dict(Caught = [], Uncaught = []), 40:dict(Caught = [], Uncaught = []), 60:dict(Caught = [], Uncaught = []), 
80:dict(Caught = [], Uncaught = []), 100:dict(Caught = [], Uncaught = [])}, 
50:{20:dict(Caught = [], Uncaught = []), 40:dict(Caught = [], Uncaught = []), 60:dict(Caught = [], Uncaught = []), 80:dict(Caught = [], Uncaught = []), 100:dict(Caught = [], Uncaught = []), 120:dict(Caught = [], Uncaught = []), 140:dict(Caught = [], Uncaught = [])} , 
75:{20:dict(Caught = [], Uncaught = []), 40:dict(Caught = [], Uncaught = []), 60:dict(Caught = [], Uncaught = []), 80:dict(Caught = [], Uncaught = []), 100:dict(Caught = [], Uncaught = [])}}


ray.init()
filbane=Path("partikkelsimulasjonar/oppdelte") 
# utfil=Path('./partiklar.pickle')
samling = []
jobbar = []

jobbar = [oppsummer.remote(fil) for fil in filbane.glob('*.pickle')]
not_ready = jobbar
while True:
    ready, not_ready = ray.wait(not_ready)
    samling.extend(ray.get(ready[0]))
    if len(not_ready)==0:
        break

# with open(Path('./partiklar.pickle'), 'rb') as fila:
#     samling = pickle.load(fila)

# bins = np.array([0.05, 0.06,0.07, 0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,12.])
bins = np.array([0.05,0.08,0.16,0.32,0.625,1.25,2.5,5,10,12])

for p in samling:
    if p['caught']:
        alle[int(p['L'])][int(p['Q'])]['Caught'].append(p['d'])
    else:
        alle[int(p['L'])][int(p['Q'])]['Uncaught'].append(p['d'])

samla =[]

for L in alle:
    for Q in alle[L]:
        caught_collected = np.asarray(alle[L][Q]['Caught'])
        uncaught_collected = np.asarray(alle[L][Q]['Uncaught'])
        caught_bins = np.asarray([np.sum((caught_collected > bins[i]) & (caught_collected < bins[i+1])) for i in range(0,len(bins)-1)])
        uncaught_bins = np.asarray([np.sum((uncaught_collected > bins[i]) & (uncaught_collected < bins[i+1])) for i in range(0,len(bins)-1)])
        alle[L][Q]['Caught_bins'] = caught_bins
        alle[L][Q]['Uncaught_bins'] = uncaught_bins
        # np.vstack((np.ones(len(caught_bins))*L,np.ones(len(caught_bins))*Q,caught_bins/(caught_bins+uncaught_bins))).T
        for b,c,u in zip(bins[1:],caught_bins,uncaught_bins):
            prosent = c/(c+u)
            samla.append([L,Q,b,prosent])
        
samla = np.asarray(samla)

samla = samla[np.invert(np.isnan(samla[:,-1]))]

fig, ax = plt.subplots()

diagram = ax.scatter(x=samla[:,1], y=samla[:,3],s= samla[:,2]*100, c=samla[:,0], label=samla[:,0])
legend1 = ax.legend(*diagram.legend_elements(), loc="lower left", title="Widths")

if not isnotebook():
    filnamn = f"caught_uncaught.png"
    fig.savefig(filnamn)
    plt.close()
# with open(Path('./caught.pickle'), 'wb') as fila:
#     pickle.dump(alle, fila)

per_QL = []

for L in alle:
    for Q in alle[L]:
        per_QL.append([L,Q,len(alle[L][Q]['Caught'])/(len(alle[L][Q]['Caught'])+len(alle[L][Q]['Uncaught']))])

per_QL = np.asarray(per_QL)

fig2, ax2 = plt.subplots()

diagram2 = ax2.scatter(x=per_QL[:,1], y=per_QL[:,2], c=per_QL[:,0], label=per_QL[:,0])
legend2 = ax2.legend(*diagram2.legend_elements(), loc="lower left", title="Widths")

if not isnotebook():
    filnamn = f"caught_uncaught_samla.png"
    fig.savefig(filnamn)
    plt.close()

#%%

per_QL_masse = []

for L in alle:
    for Q in alle[L]:
        per_QL_masse.append([L,Q,len(alle[L][Q]['Caught'])/(len(alle[L][Q]['Caught'])+len(alle[L][Q]['Uncaught']))])

per_QL_masse = np.asarray(per_QL_masse)

fig2, ax2 = plt.subplots()

diagram2 = ax2.scatter(x=per_QL[:,1], y=per_QL[:,2], c=per_QL[:,0], label=per_QL[:,0])
legend2 = ax2.legend(*diagram2.legend_elements(), loc="lower left", title="Widths")

if not isnotebook():
    filnamn = f"caught_uncaught_samla.png"
    fig.savefig(filnamn)
    plt.close()
