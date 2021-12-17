# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
import matplotlib
matplotlib.use("Agg")
# from kornfordeling import get_PSD_part
import pickle
# import random
from sti_gen import Particle, Rib, simulering
from lag_video import sti_animasjon
# import numpy as np
# from datagenerering import lag_tre, tre_objekt, lag_tre_multi
# from hjelpefunksjonar import finn_fil
# import ray
import datetime
from pathlib import Path
from subprocess import run

tal = 1000
rnd_seed=1
tider = {}

for pickle_namn in ["../TONSTAD_FOUR_Q20_FOUR TRIALONE.pickle",
"../TONSTAD_FOUR_Q20_FOUR CHECK.pickle",
"../TONSTAD_FOUR_Q20_FOUR REPEAT.pickle",
"../TONSTAD_FOUR_Q40_FOUR.pickle",
"../TONSTAD_FOUR_Q40_REPEAT.pickle",
"../TONSTAD_FOUR_Q60_FOUR.pickle",
"../TONSTAD_FOUR_Q60_FOUR REPEAT.pickle",
"../TONSTAD_FOUR_Q80_FOURDTCHANGED.pickle",
"../TONSTAD_FOUR_Q80_FOUR.pickle",
"../TONSTAD_FOUR_Q100_FOUR DT.pickle",
"../TONSTAD_FOUR_Q100_FOUR.pickle",
"../Tonstad_THREE_Q20_THREE.pickle",
"../Tonstad_THREE_Q40_THREE.pickle",
"../Tonstad_THREE_Q40_THREE_EXTRA.pickle",
"../Tonstad_THREE_Q40_THREE FINAL.pickle",
"../Tonstad_THREE_Q60_THREE.pickle",
"../Tonstad_THREE_Q80_THREE.pickle",
"../Tonstad_THREE_Q80_THREE_EXTRA.pickle",
"../Tonstad_THREE_Q80EXTRA2_THREE.pickle",
"../Tonstad_THREE_Q100_THREE.pickle",
"../Tonstad_THREE_Q100_THREE_EXTRA.pickle",
"../Tonstad_THREE_Q100_EXTRA2_THREE.pickle",
"../Tonstad_THREE_Q100_THREE_EXTRA3.pickle",
"../TONSTAD_TWO_Q20_TWO.pickle",
"../TONSTAD_TWO_Q20_TWO2.pickle",
"../TONSTAD_TWO_Q20_TWO3.pickle",
"../TONSTAD_TWO_Q40_TWO.pickle",
"../TONSTAD_TWO_Q60_TWO.pickle",
"../TONSTAD_TWO_Q80_TWO.pickle",
"../TONSTAD_TWO_Q100_TWO.pickle",
"../TONSTAD_TWO_Q120_TWO.pickle",
"../TONSTAD_TWO_Q140_TWO.pickle"]:

    pickle_fil = Path(pickle_namn)

    talstart = datetime.datetime.now()
    t_span = (0,179)
    print("Skal henta tre_objekt")
    with open(pickle_fil,'rb') as f:
        tre = pickle.load(f)
    print("Ferdig med tre-objektet")

    ribs = [Rib(rib) for rib in tre.ribs]

    sim_args = dict(fps = 20, t_span=t_span,
    linear = True, lift = True, addedmass = True, wrap_max = 50,
    method = 'BDF', atol = 1e-1, rtol = 1e-1, 
    verbose = False, collision_correction = True, hdf5_fil=pickle_fil.with_suffix(".hdf5"),  multi = False)
    laga_film = True


    # for tal in talsamling:

    partikkelfil = Path(f"./partikkelsimulasjonar/particles_{pickle_fil.stem}_{sim_args['method']}_{tal}_{sim_args['atol']:.0e}_{'linear' if sim_args['linear'] else 'NN'}.pickle")
    if not partikkelfil.exists():
        particle_list = simulering(tal, rnd_seed, tre, **sim_args)
        with open(partikkelfil, 'wb') as f:
            pickle.dump(particle_list, f)
    else:
        with open(partikkelfil, 'rb') as f:
            particle_list = pickle.load(f)

    caught = 0
    caught_mass = 0
    uncaught=0
    uncaught_mass = 0
    for pa in particle_list:
        if pa.sti_dict[round(pa.sti_dict['final_time']*100)]['caught']:
            caught += 1
            caught_mass += pa.mass
        else:
            uncaught += 1
            uncaught_mass += pa.mass

    print(pickle_fil.stem)
    print(f"Av {len(particle_list)} partiklar vart {caught} fanga, altså {100* caught/len(particle_list):.2f}%, og det er {1e3*caught_mass:.2f} mg")
    print(f"Av {len(particle_list)} partiklar vart {uncaught} ikkje fanga, altså {100* uncaught/len(particle_list):.2f}%, og det er {1e3*uncaught_mass:.2f} mg")

    if laga_film:
        start_film = datetime.datetime.now()
        film_fil = partikkelfil.with_suffix(".mp4") #Path(f"./filmar/sti_{pickle_fil.stem}_{sim_args['method']}_{len(particle_list)}_{sim_args['atol']:.0e}.mp4")
        sti_animasjon(particle_list, ribs,t_span=t_span, hdf5_fil = pickle_fil.with_suffix(".hdf5"),  utfilnamn=film_fil, fps=sim_args['fps'])
        print("Brukte  {} s på å laga film".format(datetime.datetime.now() - start_film))

        # run(f"rsync {film_fil} havrevol@login.ansatt.ntnu.no:", shell=True)


    tider[pickle_fil.stem] = dict(totalt = datetime.datetime.now() - talstart, 
        berre_film =  datetime.datetime.now() - start_film, berre_sim = start_film-talstart)
    # break

for t in tider:
    print(f"{t} brukte {tider[t]['berre_sim']} på simulering og  {tider[t]['berre_film']} på film og  {tider[t]['totalt']} på alt.")
