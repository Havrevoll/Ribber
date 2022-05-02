# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
import matplotlib
matplotlib.use("Agg")
import pickle
from sti_gen import Rib, simulering #,Particle
from lag_video import sti_animasjon
from hjelpefunksjonar import finn_fil
import datetime
from pathlib import Path
import logging
import numpy as np
import h5py
from constants import multi
from datagenerering import tre_objekt

tal = 1000
rnd_seed=1
tider = {}

pickle_filer = [
    "rib25_Q20_1",
    "rib25_Q20_2",
    "rib25_Q20_3",
    "rib25_Q40_1",
    "rib25_Q40_2",
    "rib25_Q60_1",
    "rib25_Q60_2",
    "rib25_Q80_1",
    "rib25_Q80_2",
    "rib25_Q100_1",
    "rib25_Q100_2",
    "rib75_Q20_1",
    "rib75_Q40_1",
    "rib75_Q40_2",
    "rib75_Q40_3",
    "rib75_Q60_1",
    "rib75_Q80_1",
    "rib75_Q80_2",
    "rib75_Q80_3",
    "rib75_Q100_1",
    "rib75_Q100_2",
    "rib75_Q100_3",
    "rib75_Q100_4",
    "rib50_Q20_1",
    "rib50_Q20_2",
    "rib50_Q20_3",
    "rib50_Q40_1",
    "rib50_Q60_1",
    "rib50_Q80_1",
    "rib50_Q100_1",
    "rib50_Q120_1",
    "rib50_Q140_1"
]

# logging.basicConfig(filename='simuleringar.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')


log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(funcName)s(%(lineno)d) %(message)s')

#File to log to
logFile = 'simuleringar.log'

#Setup File handler
file_handler = logging.FileHandler(logFile)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

#Setup Stream Handler (i.e. console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

#Get our logger
app_log = logging.getLogger()

#Add both Handlers
app_log.addHandler(file_handler)
app_log.addHandler(stream_handler)

app_log.setLevel(logging.DEBUG)

app_log.debug('Byrja simuleringa med denne lista:')
app_log.debug(f"{pickle_filer}")


for pickle_namn in pickle_filer:

    talstart = datetime.datetime.now()
    # pickle_namn = "TONSTAD_FOUR_Q40_REPEAT.pickle"
    # assert pickle_fil.exists()
    pickle_fil = finn_fil([Path("data").joinpath(Path(pickle_namn).with_suffix(".pickle")), Path("~/hard/").joinpath(Path(pickle_namn)).expanduser(), Path("/mnt/g/pickle/").joinpath(Path(pickle_namn))])

    assert pickle_fil.exists() and pickle_fil.with_suffix(".hdf5").exists()

    with h5py.File(pickle_fil.with_suffix(".hdf5"), 'r') as f:
        U = f.attrs['U']
        L = f.attrs['L']
        
    talstart = datetime.datetime.now()
    t_span = (0,179)

    sim_args = dict(fps = 20, t_span=t_span,
    linear = True, lift = True, addedmass = True, wrap_max = 50,
    method = 'BDF', atol = 1e-1, rtol = 1e-1, 
    verbose = False, collision_correction = True, hdf5_fil=pickle_fil.with_suffix(".hdf5"),  multi = multi, L=L, U=U)
    laga_film = False


    # for tal in talsamling:
    app_log.info(f"Byrja med {pickle_fil.stem}")

    partikkelfil = Path(f"./partikkelsimulasjonar/particles_{pickle_fil.stem}_{sim_args['method']}_{tal}_{sim_args['atol']:.0e}_{'linear' if sim_args['linear'] else 'NN'}.pickle")
    if not partikkelfil.exists():

        app_log.info("Skal henta tre.")
        with open(pickle_fil,'rb') as f:
            tre = pickle.load(f)
        app_log.info("Ferdig å henta tre.")

        ribs = [Rib(rib) for rib in tre.ribs]
        particle_list = simulering(tal, rnd_seed, tre, **sim_args)
        with open(partikkelfil, 'wb') as f:
            pickle.dump(particle_list, f)
        del tre
    else:
        app_log.info("Berekningane fanst frå før, hentar dei.")
        with open(partikkelfil, 'rb') as f:
            particle_list = pickle.load(f)
        with h5py.File(pickle_fil.with_name(f"{pickle_fil.stem}_ribs.hdf5"),'r') as f:
            ribs = [Rib(rib) for rib in np.asarray(f['ribs'])]

    # caught = 0
    # caught_mass = 0
    # uncaught=0
    # uncaught_mass = 0
    # for pa in particle_list:
    #     if pa.sti_dict[round(pa.sti_dict['final_time']*100)]['caught']:
    #         caught += 1
    #         caught_mass += pa.mass
    #     else:
    #         uncaught += 1
    #         uncaught_mass += pa.mass

    app_log.info("Brukte  {} s på å simulera.".format(datetime.datetime.now() - talstart))
    # app_log.info(f"Av {len(particle_list)} partiklar vart {caught} fanga, altså {100* caught/len(particle_list):.2f}%, og det er {1e6*caught_mass:.2f} mg")
    # app_log.info(f"Av {len(particle_list)} partiklar vart {uncaught} ikkje fanga, altså {100* uncaught/len(particle_list):.2f}%, og det er {1e6*uncaught_mass:.2f} mg")

    start_film = datetime.datetime.now()
    if laga_film:
        film_fil = partikkelfil.with_suffix(".mp4") #Path(f"./filmar/sti_{pickle_fil.stem}_{sim_args['method']}_{len(particle_list)}_{sim_args['atol']:.0e}.mp4")
        if not film_fil.exists():
            sti_animasjon(particle_list, ribs,t_span=t_span, hdf5_fil = pickle_fil.with_suffix(".hdf5"),  utfilnamn=film_fil, fps=60)
            app_log.info("Brukte  {} s på å laga film".format(datetime.datetime.now() - start_film))
        else:
            app_log.info("Filmen finst jo frå før, hoppar over dette steget.")

        # run(f"rsync {film_fil} havrevol@login.ansatt.ntnu.no:", shell=True)


    tider[pickle_fil.stem] = dict(totalt = datetime.datetime.now() - talstart, 
        berre_film =  datetime.datetime.now() - start_film, berre_sim = start_film-talstart)
    # break

for t in tider:
    app_log.info(f"{t} brukte {tider[t]['berre_sim']} på simulering og  {tider[t]['berre_film']} på film og  {tider[t]['totalt']} på alt.")
