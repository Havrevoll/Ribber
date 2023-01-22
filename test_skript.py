# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
import builtins
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
# import multiprocessing
import os
import pickle
import random
from datetime import datetime as dt
from pathlib import Path

import h5py
import numpy as np
import ray
import requests

from datagenerering import lagra_tre
from get_u_delaunay import get_u
from hjelpefunksjonar import create_bins, f2t, scale_bins
from kornfordeling import get_PSD_part
from lag_sti import lag_sti, remote_lag_sti
from lag_tre import lag_tre_multi
from lag_video import sti_animasjon
from particle import Particle
from rib import Rib

# from ray.exceptions import GetTimeoutError
# from ray.experimental.state.api import list_tasks


SIM_TIMEOUT = 1
tal = 1000
rnd_seed = 1
tider = {}
einskildpartikkel = 22
linear = lift = addedmass = True
length = 8000

pickle_filer = [
    "rib25_Q20_1",
    # #"rib25_Q20_2", "rib25_Q20_3",
    "rib25_Q40_1",
    # # "rib25_Q40_2",
    # "rib25_Q60_1",
    # # "rib25_Q60_2",
    # "rib25_Q80_1",
    # # "rib25_Q80_2",
    # "rib25_Q100_1",
    # "rib25_Q100_2",
    "rib75_Q20_1",
    "rib75_Q40_1",
    # "rib75_Q40_2", "rib75_Q40_3",
    "rib75_Q60_1", "rib75_Q80_1",
    # "rib75_Q80_2", "rib75_Q80_3",
    "rib75_Q100_1",
    # # "rib75_Q100_2", "rib75_Q100_3", "rib75_Q100_4",
    # "rib50_Q20_1",
    # # "rib50_Q20_2", "rib50_Q20_3",
    # "rib50_Q40_1",
    # "rib50_Q60_1", "rib50_Q80_1", "rib50_Q100_1", "rib50_Q120_1", "rib50_Q140_1"
    ]

graderingar = [#0.05, 
0.06, 0.07#, 0.08, 0.09, 0.1, 0.2, 0.3,
# 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,12
]

skaleringar = [1] # 40, 100, 1000]

while True:
    check_method = input("Method? [standard: RK45], vel mellom RK23, RK45, DOP853, Radau, BDF, LSODA: ").upper()
    if check_method == '':
        method = 'RK45'
        break
    elif check_method in ['RK23', 'RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']:
        method = check_method
        break
    else:
        print("Skriv ein skikkeleg metode")
# method_2nd = 'RK45'
# Denne tråden forklarer litt om korleis ein skal setja atol og rtol: https://stackoverflow.com/questions/67389644/floating-point-precision-of-scipy-solve-ivp
verbose = False
collision_correction = True
laga_film = False
while True:
    check_multi = input("Multi? ['pool','ray','no'] [default: pool]").lower()
    if check_multi in ['p', 'pool'] or check_multi == '':
        multi = 'pool'
        break
    elif check_multi in ['n','nei','no']:
        multi = 'false'
        break
    elif check_multi in ['ray', 'r']:
        multi = 'ray'
        break
    else:
        print("Ikkje eit ekte svar, prøv på nytt")


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

data_dir = Path("./data")

if not data_dir.exists():
    os.makedirs(data_dir)
else:
    assert data_dir.is_dir()

sim_dir = Path("./runs")


for namn in pickle_filer:
    hdf5_fil = data_dir.joinpath(namn).with_suffix(".hdf5")

    if not hdf5_fil.exists():
        app_log.info(f"hdf5-fila {hdf5_fil} låg ikkje inne, må lasta ned")
        nedlast_tid= dt.now()
        hdf5_fil_innhald = requests.get(f"http://folk.ntnu.no/havrevol/hdf5/{hdf5_fil.name}")
        with open(hdf5_fil,'wb') as fp:
            fp.write(hdf5_fil_innhald.content)
        app_log.info(f"Ferdig å lasta ned, brukte {dt.now()-nedlast_tid}, går vidare.")
        del nedlast_tid, fp

    for skalering in skaleringar:
        length = length*skalering
        if skalering == 1:
            pickle_namn = Path(namn).with_suffix(".pickle")
        else:
            pickle_namn = Path(namn+f"_scale{skalering}").with_suffix(".pickle")

        talstart = dt.now()
        f_span = (0,3598)
        t_span = (f2t(f_span[0],scale=skalering), f2t(f_span[0],scale=skalering))
        rtol = 1e-2
        atol = 1e-2*skalering
        tre = None

        graderingsliste = create_bins(scale_bins(np.asarray(graderingar),skalering))

        for gradering in graderingsliste:
            graderingstart = dt.now()
            pickle_fil = data_dir.joinpath(Path(pickle_namn))
            app_log.info(f"Byrja med {namn}, gradering {gradering}")

            particle_dir = sim_dir.joinpath(Path(pickle_fil.stem))
            if not particle_dir.exists():
                os.makedirs(particle_dir)

            partikkelfil = sim_dir.joinpath(pickle_fil.stem).joinpath( f"{method}_{tal}_{[round(i,3) for i in gradering]}_{skalering}_{atol:.0e}_{'linear' if linear else 'NN'}_23.11.22.pickle")
            if not partikkelfil.exists():
                if linear and tre is None:
                    app_log.info(f"Skal sjekka om treet finst som heiter {pickle_fil.name}.")
                    if pickle_fil.exists():
                        app_log.info(f"Ja, det finst, hentar det.")
                        with open(pickle_fil, 'rb') as f:
                            tre = pickle.load(f)
                        app_log.info("Ferdig å henta tre.")
                    else:
                        app_log.info(f"Det finst ikkje, må laga det.")
                        tre = lag_tre_multi((f_span[0],f_span[1]+1),filnamn_inn = hdf5_fil, skalering=skalering)
                        app_log.info(f"Ferdig å laga, skal lagra det.")
                        lagra_tre(tre,pickle_fil)
                        app_log.info(f"Ferdig å lagra det.")
                    # ribs = [Rib(rib) for rib in tre.ribs]
                    # particle_list = simulering(tal, tre, PSD=np.asarray([[gradering[0],0], [gradering[1],1]]), **sim_args)
                elif not linear and tre is None:
                    app_log.info(f"Skal berre laga kd-tre.")
                    tre = lag_tre_multi((f_span[0],f_span[1]+1),filnamn_inn = hdf5_fil, skalering=skalering, linear=False)
                    app_log.info(f"Ferdig å laga kd-tre.")


                with h5py.File(data_dir.joinpath(namn+"_ribs").with_suffix(".hdf5"), 'w') as f:
                    f.create_dataset("ribs", data=np.asarray(tre.ribs))
                start = dt.now()
                ribs = [Rib(rib, µ=(0.85 if rib_index < 2 else 1.5)) for rib_index, rib in enumerate(tre.ribs)]

                with h5py.File(hdf5_fil, 'r') as f:
                    max_y = np.max(np.asarray(f['y'])*skalering)
                del f

                # Her blir partiklane laga:
                random.seed(rnd_seed)
                diameters = get_PSD_part(tal, PSD=np.asarray([[gradering[0], 0], [gradering[1], 1]]), rnd_seed=rnd_seed).tolist()
                particle_list = [Particle(diameter=float(d), init_position=[ribs[0].get_rib_middle()[0], random.uniform(ribs[0].get_rib_middle()[1]+ribs[0].get_rib_dimensions()[0], max_y), 0, 0], init_time = random.randrange(0, 1000 )) for d in diameters]

                for i, p in enumerate(particle_list):
                    p.atol, p.rtol = atol, rtol
                    p.method = method
                    p.linear, p.lift, p.addedmass = linear, lift, addedmass
                    # rettar opp init_tid slik at det blir eit tal som finst i datasettet.
                    # p.init_time = floor(p.init_time * fps)/fps
                    p.index = i
                    p.resting_tolerance = 0.0001 if method == "BDF" else 0.01
                    p.scale = skalering
                    p.length = length
                del i,p

                builtins.tre = tre

                if multi == 'ray': 
                    ray.init(local_mode=False,include_dashboard=True, num_cpus=8)  # dashboard_port=8266,),num_cpus=4
                    # tre_plasma = ray.put(tre)

                    lag_sti_args = dict(ribs =ribs, f_span=f_span, get_u=get_u, skalering=skalering, 
                                            verbose=verbose, collision_correction=collision_correction)



                    # index_list = {pa.index:{'job':remote_lag_sti.remote(particle=pa, **lag_sti_args),'particle':pa} for pa in particle_list} #index som key, job og particle i ein dict under der
                    # job_list_strings = {index_list[i]['job'].task_id().hex():i for i in index_list} # task-id som string som key, index som value
                    # # task_liste = list_tasks(filters=[("scheduling_state", "!=", "SCHEDULED")]) # lista over dei som er running, som string
                    # running = {job_list_strings[p['task_id']]:dt.now() for p in list_tasks(filters=[("scheduling_state", "!=", "SCHEDULED")])} #index som key, tid for oppstart som value
                    # scheduled = [job_list_strings[p['task_id']] for p in list_tasks(filters=[("scheduling_state", "=", "SCHEDULED")])] # berre ei liste med index som er scheduled. Maks 100

                    # cancelled = []

                    # while len(running) > 0:
                    #     ready, _ = ray.wait([index_list[i]['job'] for i in running.keys()], timeout=.1)

                    #     if len(ready) > 0:
                    #         elem = job_list_strings[ready[0].task_id().hex()]
                    #     else:
                    #         elem = min(running,key=running.get)
                    #     tid = running.pop(elem)
                    #     app_log.info(f"skal sjekka partikkel {elem}, gått i {(dt.now()-tid).seconds} sekund, dei som no er att er {[(k,(dt.now()-v).seconds) for k,v in running.items()]}")

                    #     if (dt.now() - tid).seconds > SIM_TIMEOUT or len(ready) > 0:
                    #         try:
                    #             app_log.info(f"Har kome til partikkel nr. {elem}")
                    #             sti_dict = ray.get(index_list[elem]['job'], timeout=(1))

                    #             assert all([i in sti_dict for i in range(sti_dict['init_time'], sti_dict['final_time']+1)]), f"Partikkel nr. {elem} er ufullstendig"
                    #             index_list[elem]['particle'].sti_dict = sti_dict

                    #         except (GetTimeoutError, AssertionError):
                    #             ray.cancel(index_list[elem]['job'], force=True)
                    #             app_log.info(f"Måtte kansellera nr. {elem}, vart visst aldri ferdig.")
                    #             cancelled.append(elem)
                    #             index_list[elem]['particle'].method = method_2nd
                    #     else:
                    #         running[elem] = tid

                    #     new_running = dict.fromkeys([job_list_strings[p['task_id']] for p in list_tasks(filters=[("scheduling_state", "!=", "SCHEDULED")]) if (job_list_strings[p['task_id']] not in running) and (job_list_strings[p['task_id']] not in cancelled)],(dt.now()))
                    #     running.update(new_running)
                    #     scheduled = [job_list_strings[p['task_id']] for p in list_tasks(filters=[("scheduling_state", "=", "SCHEDULED")])]

                    # if len(cancelled) > 0:
                    #     app_log.info("Skal ta dei som ikkje klarte BDF")
                    index_list =  {pa.index:{'particle':pa} for pa in particle_list}
                    not_ready = []
                    jobs = {}
                    for i in [p.index for p in particle_list]:
                        index_list[i]['job'] = remote_lag_sti.remote(particle=index_list[i]['particle'], **lag_sti_args)
                        not_ready.append(index_list[i]['job'])
                        jobs[index_list[i]['job']] = i
                    del i
                    while True:
                        ready, not_ready = ray.wait(not_ready)
                        sti_dict = ray.get(ready[0])
                        app_log.info(f"Fekk nr. {jobs[ready[0]]} som var klar.")
                        # ny_sti_dict = deepcopy_sti_dict(sti_dict)

                        assert all([i in sti_dict for i in range(sti_dict['init_time'], sti_dict['final_time']+1)]), f"Partikkel nr. {jobs[ready[0]]} er ufullstendig"

                        index_list[jobs[ready[0]]]['particle'].sti_dict = sti_dict
                        index_list[jobs.pop(ready[0])].pop('job')

                        app_log.info(f"Dei som står att no er {[index_list[jobs[p]]['particle'].index for p in not_ready] if len(not_ready)<100 else len(not_ready)}")
                        if len(not_ready) == 0:
                            break

                elif multi == 'pool':
                    # with multiprocessing.Pool(processes = 8) as p:
                    #     stiar = p.starmap(lag_sti, [(ribs, f_span, particle, get_u) for particle in particle_list] )
                        
                    # for sti,pa in zip(stiar,particle_list):
                    #     pa.sti_dict = sti

                    with ProcessPoolExecutor(max_workers=8) as executor:
                        stiar = {executor.submit(lag_sti, ribs, f_span, particle, get_u):particle for particle in particle_list}

                    for sti in as_completed(stiar):
                        pa = stiar[sti]
                        pa.sti_dict = sti.result()

                else:
                    lag_sti_args = dict(ribs =ribs, f_span=f_span, get_u=get_u, skalering=skalering,
                                            verbose=verbose, collision_correction=collision_correction)

                    for pa in particle_list:
                        # if pa.index==einskildpartikkel:
                            pa.sti_dict = lag_sti(particle = pa, **lag_sti_args)
                            assert all([i in pa.sti_dict for i in range(pa.sti_dict['init_time'], pa.sti_dict['final_time']+1)]), f"Partikkel nr. {pa.index} er ufullstendig"

                for pa in particle_list:
                    if not hasattr(pa,'sti_dict'):
                        # try:
                            app_log.info(f"Hadde ikkje fått sti_dict frå {pa.index}. Prøver å henta den no.")
                            sti_dict = ray.get(index_list[pa.index]['job'], timeout=(5))

                            assert all([i in sti_dict for i in range(sti_dict['init_time'], sti_dict['final_time']+1)]), f"Partikkel nr. {pa.index} er ufullstendig"
                            pa.sti_dict = sti_dict
                        # except (GetTimeoutError, AssertionError):
                        #     ray.cancel(index_list[elem]['job'], force=True)
                        #     app_log.info(f"Måtte kansellera nr. {elem}, vart visst aldri ferdig.")

                    assert hasattr(pa,'sti_dict'), f"problem i {pa.index}"
                ray.shutdown()
                with open(partikkelfil, 'wb') as f:
                    pickle.dump(particle_list, f)
                app_log.info(f"Lagra partiklane som {partikkelfil}")
                # del tre
            elif laga_film:
                app_log.info("Berekningane fanst frå før, hentar dei.")
                with open(partikkelfil, 'rb') as f:
                    particle_list = pickle.load(f)
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

            app_log.info(f"Brukte  {dt.now() - graderingstart} s på å simulera. Til saman er brukt {dt.now()-talstart}.")
            # app_log.info(f"Av {len(particle_list)} partiklar vart {caught} fanga, altså {100* caught/len(particle_list):.2f}%, og det er {1e6*caught_mass:.2f} mg")
            # app_log.info(f"Av {len(particle_list)} partiklar vart {uncaught} ikkje fanga, altså {100* uncaught/len(particle_list):.2f}%, og det er {1e6*uncaught_mass:.2f} mg")

            start_film = dt.now()
            if laga_film:
                # Path(f"./filmar/sti_{pickle_fil.stem}_{sim_args['method']}_{len(particle_list)}_{sim_args['atol']:.0e}.mp4")
                film_fil = partikkelfil.with_suffix(".mp4")
                if not film_fil.exists():
                    sti_animasjon(particle_list, ribs, t_span=t_span, hdf5_fil=hdf5_fil, utfilnamn=film_fil, fps=60, skalering=skalering)
                    app_log.info("Brukte  {} s på å laga film".format(
                        dt.now() - start_film))
                else:
                    app_log.info(
                        "Filmen finst jo frå før, hoppar over dette steget.")

            tider[pickle_fil.stem] = dict(totalt=dt.now() - talstart,
                                        berre_film=dt.now() - start_film, berre_sim=start_film-talstart)
            # break

        for t in tider:
            app_log.info(
                f"{t} brukte {tider[t]['berre_sim']} på simulering og  {tider[t]['berre_film']} på film og  {tider[t]['totalt']} på alt.")
