# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
from lag_sti import lag_sti, remote_lag_sti  # ,Particle
from ray.exceptions import GetTimeoutError
import ray
import random
from kornfordeling import get_PSD_part
from particle import Particle
from rib import Rib
import h5py
import numpy as np
import logging
from pathlib import Path
import datetime
from hjelpefunksjonar import finn_fil
from lag_video import sti_animasjon
import pickle
from math import floor
import matplotlib
matplotlib.use("Agg")

tal = 200
rnd_seed = 1
tider = {}

pickle_filer = [
    # "rib25_Q20_1", 
    # "rib25_Q20_2", "rib25_Q20_3", 
    # "rib25_Q40_1", 
    # "rib25_Q40_2", 
    # "rib25_Q60_1", 
    # "rib25_Q60_2", 
    # "rib25_Q80_1", 
    # "rib25_Q80_2", 
    # "rib25_Q100_1", 
    # "rib25_Q100_2", 
    # "rib75_Q20_1", "rib75_Q40_1", 
    # "rib75_Q40_2", "rib75_Q40_3", 
    # "rib75_Q60_1", "rib75_Q80_1", 
    # "rib75_Q80_2", "rib75_Q80_3", 
    # "rib75_Q100_1", 
    # "rib75_Q100_2", "rib75_Q100_3", "rib75_Q100_4", 
    # "rib50_Q20_1", 
    # "rib50_Q20_2", "rib50_Q20_3", 
    "rib50_Q40_1", 
    # "rib50_Q60_1", "rib50_Q80_1", "rib50_Q100_1", "rib50_Q120_1", "rib50_Q140_1"
    ]
pickle_filer = ["rib50_Q40_1_skalert" ]

graderingsliste = [
    [10, 12], 
    [9, 10],
    [8, 9],
    [7, 8],
    [6, 7],
    [5, 6],
    [4, 5],[3, 4],[2, 3],[1, 2],[0.9, 1],[0.8, 0.9],
                    [0.7, 0.8],[0.6, 0.7],[0.5, 0.6],[0.4, 0.5],[0.3, 0.4],[0.2, 0.3],[0.1, 0.2],[0.09, 0.1],
                    [0.08, 0.09],[0.07, 0.08],[0.06, 0.07],
                    [0.05, 0.06]]


temp = np.array([0.11, 0.13, 0.15, 0.18, 0.21, 0.24, 0.61, 1.27, 2.21, 3.50, 5.02, 6.75, 8.60, 10.64, 12.89, 33.56, 54.38, 74.98, 95.44, 118.31, 140.16, 160.38, 179.30, 197.13])
graderingsliste = np.vstack((temp[:-1],temp[1:])).T 
del temp

# logging.basicConfig(filename='simuleringar.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')


log_formatter = logging.Formatter(
    '%(asctime)s %(levelname)s %(name)s %(funcName)s(%(lineno)d) %(message)s')

# File to log to
logFile = 'simuleringar.log'

# Setup File handler
file_handler = logging.FileHandler(logFile)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

# Setup Stream Handler (i.e. console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

# Get our logger
app_log = logging.getLogger()

# Add both Handlers
app_log.addHandler(file_handler)
app_log.addHandler(stream_handler)

app_log.setLevel(logging.DEBUG)

app_log.debug('Byrja simuleringa med denne lista:')
app_log.debug(f"{pickle_filer}")


for namn in pickle_filer:
    pickle_namn = Path(namn).with_suffix(".pickle")
    hdf5_fil = Path("data").joinpath(namn).with_suffix(".hdf5")

    # pickle_namn = "TONSTAD_FOUR_Q40_REPEAT.pickle"
    # assert pickle_fil.exists()
    pickle_fil = finn_fil([Path("data").joinpath(Path(pickle_namn)), Path("~/hard/").joinpath(
        Path(pickle_namn)).expanduser(), Path("/mnt/g/pickle/").joinpath(Path(pickle_namn))])

    assert pickle_fil.exists()
    assert hdf5_fil.exists()

    talstart = datetime.datetime.now()
    t_span = (0, 179)

    # sim_args = dict(fps = 20, t_span=t_span, linear = True, lift = True, addedmass = True, wrap_max = 50, method = 'BDF', atol = 1e-1, rtol = 1e-1, verbose = False, collision_correction = True, hdf5_fil=pickle_fil.with_suffix(".hdf5"),  multi = multi)
    fps = 20
    linear = True
    lift = True
    addedmass = True
    wrap_max = 50
    method = 'BDF'
    atol = 1e-1
    rtol = 1e-1
    verbose = False
    collision_correction = True
    laga_film = True
    multi = True

    for gradering in graderingsliste:
        app_log.info(f"Byrja med {pickle_fil.stem}, gradering {gradering}")

        partikkelfil = Path( f"./partikkelsimulasjonar/particles_{pickle_fil.stem}_{method}_{tal}_{gradering}_{atol:.0e}_{'linear' if linear else 'NN'}.pickle")
        if not partikkelfil.exists():

            app_log.info("Skal henta tre.")
            with open(pickle_fil, 'rb') as f:
                tre = pickle.load(f)
            app_log.info("Ferdig å henta tre.")

            ribs = [Rib(rib) for rib in tre.ribs]
            # particle_list = simulering(tal, tre, PSD=np.asarray([[gradering[0],0], [gradering[1],1]]), **sim_args)

            random.seed(rnd_seed)

            start = datetime.datetime.now()
            ribs = [Rib(rib, µ=0.5 if rib_index < 2 else 1)
                    for rib_index, rib in enumerate(tre.ribs)]

            with h5py.File(hdf5_fil, 'r') as f:
                max_y = np.max(np.asarray(f['y']))

            diameters = get_PSD_part(tal, PSD=np.asarray(
                [[gradering[0], 0], [gradering[1], 1]]), rnd_seed=rnd_seed).tolist()
            particle_list = [Particle(float(d), [ribs[0].get_rib_middle()[0], random.uniform(
                ribs[0].get_rib_middle()[1]+8, max_y), 0, 0], random.uniform(0, 50)) for d in diameters]

            for i, p in enumerate(particle_list):
                p.atol, p.rtol = atol, rtol
                p.method = method
                p.linear, p.lift, p.addedmass = linear, lift, addedmass
                # rettar opp init_tid slik at det blir eit tal som finst i datasettet.
                p.init_time = floor(p.init_time * fps)/fps
                p.index = i
                p.resting_tolerance = 0.0001 if method == "BDF" else 0.01

            if multi:
                ray.init()  # dashboard_port=8266,num_cpus=4)
                tre_plasma = ray.put(tre)
                jobs = {remote_lag_sti.remote(ribs, t_span, particle=pa, tre=tre_plasma, fps=fps, wrap_max=wrap_max,
                                              verbose=verbose, collision_correction=collision_correction): pa for pa in particle_list}

                not_ready = list(jobs.keys())
                cancelled = []
                for elem in not_ready:
                    # if len(ready) == 0:
                    try:


                        sti_dict = ray.get(elem, timeout=(20))

                        assert all([i in sti_dict.keys() for i in np.linspace(round(sti_dict['init_time']*100), round(sti_dict['final_time']*100), round(
                            (sti_dict['final_time']-sti_dict['init_time'])*20)+1).astype(int)]), f"Partikkel nr. {jobs[elem].index} har ein feil i seg, ikkje alle elementa er der"
                        jobs[elem].sti_dict = sti_dict
                        # jobs[ready[0]].sti_dict = sti_dict
                        app_log.info(
                            f"Har kome til partikkel nr. {jobs[elem].index}")
                    except (GetTimeoutError, AssertionError):
                        ray.cancel(elem, force=True)
                        app_log.info(
                            f"Måtte kansellera {jobs[elem].index}, vart visst aldri ferdig.")
                        cancelled.append(jobs[elem])
                        jobs[elem].method = "RK23"

                if len(cancelled) > 0:
                    app_log.info("Skal ta dei som ikkje klarte BDF")
                    jobs2 = {remote_lag_sti.remote(ribs, t_span, particle=pa, tre=tre_plasma, fps=fps, wrap_max=wrap_max,
                                                    verbose=verbose, collision_correction=collision_correction): pa for pa in cancelled} #var cancelled
                    not_ready = list(jobs2.keys())
                    while True:
                        ready, not_ready = ray.wait(not_ready)
                        sti_dict = ray.get(ready[0])
                        assert all([i in sti_dict.keys() for i in np.linspace(round(sti_dict['init_time']*100), round(sti_dict['final_time']*100), round(
                            (sti_dict['final_time']-sti_dict['init_time'])*20)+1).astype(int)]), f"Partikkel nr. {jobs2[ready[0]].index} har ein feil i seg, ikkje alle elementa er der"
                        jobs2[ready[0]].sti_dict = sti_dict

                        app_log.info(
                            f"Dei som står att no er {[jobs2[p].index for p in not_ready] if len(not_ready)<100 else len(not_ready)}")
                        if len(not_ready) == 0:
                            break

                ray.shutdown()
            else:
                for pa in particle_list:
                    if pa.index == 19:
                        pa.sti_dict = lag_sti(ribs, t_span, particle=pa, tre=tre, fps=fps, wrap_max=wrap_max,
                                              verbose=verbose, collision_correction=collision_correction)
                        assert all([i in pa.sti_dict.keys() for i in np.linspace(round(pa.sti_dict['init_time']*100), round(pa.sti_dict['final_time']*100), round(
                            (pa.sti_dict['final_time']-pa.sti_dict['init_time'])*20)+1).astype(int)]), f"Partikkel nr. {pa.index} har ein feil i seg, ikkje alle elementa er der"

            for pa in particle_list:
                assert hasattr(pa,'sti_dict')
            with open(partikkelfil, 'wb') as f:
                pickle.dump(particle_list, f)
            del tre
        else:
            app_log.info("Berekningane fanst frå før, hentar dei.")
            with open(partikkelfil, 'rb') as f:
                particle_list = pickle.load(f)
            # with h5py.File(pickle_fil.with_name(f"{pickle_fil.stem}_ribs.hdf5"), 'r') as f:
            #     ribs = [Rib(rib) for rib in np.asarray(f['ribs'])]
            with open(pickle_fil, 'rb') as f:
                ribs =  [Rib(rib) for rib in pickle.load(f).ribs]
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

        app_log.info("Brukte  {} s på å simulera.".format(
            datetime.datetime.now() - talstart))
        # app_log.info(f"Av {len(particle_list)} partiklar vart {caught} fanga, altså {100* caught/len(particle_list):.2f}%, og det er {1e6*caught_mass:.2f} mg")
        # app_log.info(f"Av {len(particle_list)} partiklar vart {uncaught} ikkje fanga, altså {100* uncaught/len(particle_list):.2f}%, og det er {1e6*uncaught_mass:.2f} mg")

        start_film = datetime.datetime.now()
        if laga_film:
            # Path(f"./filmar/sti_{pickle_fil.stem}_{sim_args['method']}_{len(particle_list)}_{sim_args['atol']:.0e}.mp4")
            film_fil = partikkelfil.with_suffix(".mp4")
            if not film_fil.exists():
                sti_animasjon(particle_list, ribs, t_span=t_span, hdf5_fil=pickle_fil.with_suffix(
                    ".hdf5"),  utfilnamn=film_fil, fps=60)
                app_log.info("Brukte  {} s på å laga film".format(
                    datetime.datetime.now() - start_film))
            else:
                app_log.info(
                    "Filmen finst jo frå før, hoppar over dette steget.")

            # run(f"rsync {film_fil} havrevol@login.ansatt.ntnu.no:", shell=True)

        tider[pickle_fil.stem] = dict(totalt=datetime.datetime.now() - talstart,
                                      berre_film=datetime.datetime.now() - start_film, berre_sim=start_film-talstart)
        # break

    for t in tider:
        app_log.info(
            f"{t} brukte {tider[t]['berre_sim']} på simulering og  {tider[t]['berre_film']} på film og  {tider[t]['totalt']} på alt.")
