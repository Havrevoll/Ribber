# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
from datetime import datetime as dt
import logging
import os
import pickle
import random
from pathlib import Path
import requests

import h5py
import numpy as np
import ray
from ray.exceptions import GetTimeoutError
from ray.experimental.state.api import list_tasks

from datagenerering import lagra_tre
from hjelpefunksjonar import create_bins, f2t, scale_bins
from kornfordeling import get_PSD_part
from lag_sti import lag_sti, remote_lag_sti
from lag_tre import lag_tre_multi
from lag_video import sti_animasjon
from particle import Particle
from rib import Rib
from get_u_analytic import get_u


tal = 1
rnd_seed = 1
tider = {}

SIM_TIMEOUT = 120

graderingar = [0.05, 0.06#, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3,
#0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,12
]

skaleringar = [1] # 40, 100, 1000]
linear = False
lift = False
addedmass = False
wrap_max = 50
method = 'RK45'
method_2nd = 'RK23'
# Denne tråden forklarer litt om korleis ein skal setja atol og rtol: https://stackoverflow.com/questions/67389644/floating-point-precision-of-scipy-solve-ivp
verbose = False
collision_correction = True
laga_film = False
multi = False

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

data_dir = Path("./data")

if not data_dir.exists():
    os.makedirs(data_dir)
else:
    assert data_dir.is_dir()

sim_dir = Path("./runs")

talstart = dt.now()
f_span = (0,3598)
t_span = (f2t(f_span[0]), f2t(f_span[0]))
rtol = 1e-1
atol = 1e-1
tre = None

# sim_args = dict(fps = 20, t_span=t_span, linear = True, lift = True, addedmass = True, wrap_max = 50, method = 'BDF', atol = 1e-1, rtol = 1e-1, verbose = False, collision_correction = True, hdf5_fil=pickle_fil.with_suffix(".hdf5"),  multi = multi)
# fps = 20/sqrt(skalering)

graderingsliste = create_bins(scale_bins(np.asarray(graderingar)))

for gradering in graderingsliste:
    graderingstart = dt.now()
    app_log.info(f"Byrja med gradering {gradering}")

    particle_dir = sim_dir.joinpath(Path("analytic"))
    if not particle_dir.exists():
        os.makedirs(particle_dir)

    partikkelfil = sim_dir.joinpath("analytisk").joinpath( f"{method}_{method_2nd}_{tal}_{[round(i,3) for i in gradering]}_{atol:.0e}_{'linear' if linear else 'NN'}_test16.9.22.pickle")
    if not partikkelfil.exists():
        
        random.seed(rnd_seed)

        start = dt.now()
        # ribs = [Rib(rib, µ=0.85 if rib_index < 2 else 1)
        #         for rib_index, rib in enumerate(tre.ribs)]

        max_y = 60

        # Her blir partiklane laga:
        diameters = get_PSD_part(tal, PSD=np.asarray([[gradering[0], 0], [gradering[1], 1]]), rnd_seed=rnd_seed).tolist()
        particle_list = [Particle(diameter=float(d), init_position=[-50, random.uniform(0, max_y), 0, 0], init_time = 0) for d in diameters]

        for i, p in enumerate(particle_list):
            p.atol, p.rtol = atol, rtol
            p.method = method
            p.linear, p.lift, p.addedmass = linear, lift, addedmass
            # rettar opp init_tid slik at det blir eit tal som finst i datasettet.
            # p.init_time = floor(p.init_time * fps)/fps
            p.index = i
            p.resting_tolerance = 0.0001 if method == "BDF" else 0.01
            p.scale = 1
            p.wrap_max = wrap_max
        del i,p

        # particle_list = particle_list[:100]
        if multi:
            ray.init(local_mode=False,include_dashboard=True, num_cpus=6)  # dashboard_port=8266,),num_cpus=4
            lag_sti_args = dict(f_span=f_span, get_u=get_u, skalering=1, wrap_max=wrap_max,
                                    verbose=verbose, collision_correction=collision_correction)



            index_list = {pa.index:{'job':remote_lag_sti.remote(particle=pa, **lag_sti_args),'particle':pa} for pa in particle_list} #index som key, job og particle i ein dict under der
            job_list_strings = {index_list[i]['job'].task_id().hex():i for i in index_list} # task-id som string som key, index som value
            task_liste = list_tasks(filters=[("scheduling_state", "!=", "SCHEDULED")]) # lista over dei som er running, som string
            running = {job_list_strings[p['task_id']]:dt.now() for p in list_tasks(filters=[("scheduling_state", "!=", "SCHEDULED")])} #index som key, tid for oppstart som value
            scheduled = [job_list_strings[p['task_id']] for p in list_tasks(filters=[("scheduling_state", "=", "SCHEDULED")])] # berre ei liste med index som er scheduled. Maks 100

            cancelled = []

            while len(running) > 0:
                ready, _ = ray.wait([index_list[i]['job'] for i in running.keys()], timeout=0.5)

                if len(ready) > 0:
                    elem = job_list_strings[ready[0].task_id().hex()]
                else:
                    elem = min(running,key=running.get)
                tid = running.pop(elem)
                app_log.info(f"skal sjekka partikkel {elem}, gått i {(dt.now()-tid).seconds} sekund, dei som no er att er {[(k,(dt.now()-v).seconds) for k,v in running.items()]}")

                if (dt.now() - tid).seconds > SIM_TIMEOUT or len(ready) > 0:
                    try:
                        app_log.info(f"Har kome til partikkel nr. {elem}")
                        sti_dict = ray.get(index_list[elem]['job'], timeout=(1))

                        assert all([i in sti_dict for i in range(sti_dict['init_time'], sti_dict['final_time']+1)]), f"Partikkel nr. {elem} er ufullstendig"
                        index_list[elem]['particle'].sti_dict = sti_dict

                    except (GetTimeoutError, AssertionError):
                        ray.cancel(index_list[elem]['job'], force=True)
                        app_log.info(f"Måtte kansellera nr. {elem}, vart visst aldri ferdig.")
                        cancelled.append(elem)
                        index_list[elem]['particle'].method = method_2nd
                else:
                    running[elem] = tid

                new_running = dict.fromkeys([job_list_strings[p['task_id']] for p in list_tasks(filters=[("scheduling_state", "!=", "SCHEDULED")]) if (job_list_strings[p['task_id']] not in running) and (job_list_strings[p['task_id']] not in cancelled)],(dt.now()))
                running.update(new_running)
                scheduled = [job_list_strings[p['task_id']] for p in list_tasks(filters=[("scheduling_state", "=", "SCHEDULED")])]

            if len(cancelled) > 0:
                app_log.info("Skal ta dei som ikkje klarte BDF")
                not_ready = []
                jobs = {}
                for i in cancelled:
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

        else:
            lag_sti_args = dict(f_span=f_span, get_u=get_u, skalering=1, wrap_max=wrap_max,
                                    verbose=verbose, collision_correction=collision_correction)
            for pa in particle_list:
                # if pa.index == 0:
                    pa.sti_dict = lag_sti(particle = pa, **lag_sti_args)
                    assert all([i in pa.sti_dict for i in range(pa.sti_dict['init_time'], pa.sti_dict['final_time']+1)]), f"Partikkel nr. {pa.index} er ufullstendig"

        for pa in particle_list:
            if not hasattr(pa,'sti_dict'):
                # try:
                    app_log.info(f"Hadde ikkje fått sti_dict frå {pa.index}. Prøver å henta den no.")
                    sti_dict = ray.get(index_list[pa.index]['job'], timeout=(5))

                    assert all([i in sti_dict for i in range(sti_dict['init_time'], sti_dict['final_time']+1)]), f"Partikkel nr. {elem} er ufullstendig"
                    pa.sti_dict = sti_dict
                # except (GetTimeoutError, AssertionError):
                #     ray.cancel(index_list[elem]['job'], force=True)
                #     app_log.info(f"Måtte kansellera nr. {elem}, vart visst aldri ferdig.")

            assert hasattr(pa,'sti_dict'), f"problem i {pa.index}"
        ray.shutdown()
        with open(partikkelfil, 'wb') as f:
            pickle.dump(particle_list, f)
        # del tre
    elif laga_film:
        app_log.info("Berekningane fanst frå før, hentar dei.")
        with open(partikkelfil, 'rb') as f:
            particle_list = pickle.load(f)

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
    # if laga_film:
    #     # Path(f"./filmar/sti_{pickle_fil.stem}_{sim_args['method']}_{len(particle_list)}_{sim_args['atol']:.0e}.mp4")
    #     film_fil = partikkelfil.with_suffix(".mp4")
    #     if not film_fil.exists():
    #         sti_animasjon(particle_list, ribs, t_span=t_span, hdf5_fil=hdf5_fil, utfilnamn=film_fil, fps=60, skalering=skalering)
    #         app_log.info("Brukte  {} s på å laga film".format(
    #             dt.now() - start_film))
    #     else:
    #         app_log.info(
    #             "Filmen finst jo frå før, hoppar over dette steget.")

    # tider[pickle_fil.stem] = dict(totalt=dt.now() - talstart,
    #                             berre_film=dt.now() - start_film, berre_sim=start_film-talstart)
    # break

for t in tider:
    app_log.info(
        f"{t} brukte {tider[t]['berre_sim']} på simulering og  {tider[t]['berre_film']} på film og  {tider[t]['totalt']} på alt.")
