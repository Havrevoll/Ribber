# -*- coding: utf-8 -*-
'''køyr funksjonar som plottingar(fil['vassføringar'])'''

import h5py
import matplotlib
# from ray.core.generated.common_pb2 import _TASKSPEC_OVERRIDEENVIRONMENTVARIABLESENTRY
matplotlib.use("Agg")

from kornfordeling import get_PSD_part
import numpy as np
# from scipy import interpolate
from scipy.integrate import solve_ivp  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#r179348322575-1
# from hjelpefunksjonar import norm, sortClockwise, finn_fil
from f import f
from rib import Rib
from particle import Particle
from pathlib import Path
from check_collision import check_all_collisions
from constants import ε

import ray
from ray.exceptions import GetTimeoutError
import datetime
from math import floor, hypot, ceil
import random
# from scipy.sparse.construct import rand #, atan2
# import psutil
import logging
from loguru import logger

app_log = logging.getLogger(__name__)

# fil = h5py.File("D:/Tonstad/alle.hdf5", 'a')
# x = np.array(h5py.File(filnamn, 'r')['x']).reshape(127,126)[ranges()]
# vass = fil['vassføringar']

from constants import collision_restitution

@logger.catch
def simulering(tal, rnd_seed, tre, fps = 20, t_span = (0,179), linear = True,  lift = True, addedmass = True, wrap_max = 50, atol = 1e-1, rtol = 1e-1, 
    method = 'RK23', laga_film = False, verbose=True, collision_correction=True, hdf5_fil=Path("./"), multi = True, L=50, U=1):
    random.seed(rnd_seed)

    start = datetime.datetime.now()
    ribs = [Rib(rib, µ = 0.5 if rib_index < 2 else 1) for rib_index, rib in enumerate(tre.ribs)]

    with h5py.File(hdf5_fil, 'r') as f:
        max_y = np.max(np.asarray(f['y'])/L)

    diameters = get_PSD_part(tal, rnd_seed=rnd_seed).tolist()
    particle_list = [Particle(float(d), [ribs[0].get_rib_middle()[0],random.uniform(ribs[0].get_rib_middle()[1]+8,max_y), 0, 0], random.uniform(0,50)) for d in diameters]

    for i, p in enumerate(particle_list):
        p.atol , p.rtol = atol, rtol
        p.method = method
        p.linear, p.lift, p.addedmass = linear, lift, addedmass
        p.init_time = floor(p.init_time *fps)/fps #rettar opp init_tid slik at det blir eit tal som finst i datasettet.
        p.index = i
        p.resting_tolerance = 0.0001 if method == "BDF" else 0.01

    
    if multi:
        ray.init()#dashboard_port=8266,num_cpus=4) 
        tre_plasma = ray.put(tre)
        jobs = {remote_lag_sti.remote(ribs, t_span, particle=pa, tre=tre_plasma, fps = fps, wrap_max=wrap_max, verbose=verbose, collision_correction=collision_correction, L=L, U=U):pa for pa in particle_list}

        not_ready = list(jobs.keys())
        cancelled = []
        for elem in not_ready:
            # if len(ready) == 0:
            try:
                
                sti_dict = ray.get(elem, timeout=(40))
                        
                assert all([i in sti_dict.keys() for i in np.linspace(round(sti_dict['init_time']*100), round(sti_dict['final_time']*100), round((sti_dict['final_time']-sti_dict['init_time'])*20)+1).astype(int)]), f"Partikkel nr. {jobs[elem].index} har ein feil i seg, ikkje alle elementa er der"
                jobs[elem].sti_dict = sti_dict
                # jobs[ready[0]].sti_dict = sti_dict
                app_log.info(f"Har kome til partikkel nr. {jobs[elem].index}")
            except (GetTimeoutError,AssertionError):
                ray.cancel(elem, force=True)
                app_log.info(f"Måtte kansellera {jobs[elem].index}, vart visst aldri ferdig.")
                cancelled.append(jobs[elem])
                jobs[elem].method = "RK23"
                
        if len(cancelled) > 0:
            app_log.debug("Skal ta dei som ikkje klarte BDF")
            jobs2 = {remote_lag_sti.remote(ribs, t_span, particle=pa, tre=tre_plasma, fps = fps, wrap_max=wrap_max, verbose=verbose, collision_correction=collision_correction):pa for pa in cancelled}
            not_ready = list(jobs2.keys())
            while True:
                ready, not_ready = ray.wait(not_ready)
                sti_dict =  ray.get(ready[0])
                assert all([i in sti_dict.keys() for i in np.linspace(round(sti_dict['init_time']*100), round(sti_dict['final_time']*100), round((sti_dict['final_time']-sti_dict['init_time'])*20)+1).astype(int)]), f"Partikkel nr. {jobs[ready[0]].index} har ein feil i seg, ikkje alle elementa er der"
                jobs2[ready[0]].sti_dict = sti_dict

                app_log.info(f"Dei som står att no er {[jobs2[p].index for p in not_ready] if len(not_ready)<100 else len(not_ready)}")
                if len(not_ready)==0:
                    break
            

        ray.shutdown()
    else:
        for pa in particle_list:
            if pa.index == 446:
                pa.sti_dict = lag_sti(ribs, t_span, particle=pa, tre=tre, fps = fps, wrap_max=wrap_max, verbose=verbose, collision_correction=collision_correction)
                assert all([i in pa.sti_dict.keys() for i in np.linspace(round(pa.sti_dict['init_time']*100), round(pa.sti_dict['final_time']*100), round((pa.sti_dict['final_time']-pa.sti_dict['init_time'])*20)+1).astype(int)]), f"Partikkel nr. {pa.index} har ein feil i seg, ikkje alle elementa er der"

    return particle_list


@ray.remote
def remote_lag_sti(ribs, t_span, particle, tre, fps=20, wrap_max = 0, verbose=True, collision_correction=True, L=50, U=1):
    return lag_sti(ribs, t_span, particle, tre, fps=fps, wrap_max = wrap_max, verbose=verbose, collision_correction=collision_correction, L=L, U=U)

def lag_sti(ribs, t_span, particle, tre, fps=20, wrap_max = 0, verbose=True, collision_correction=True, L=50, U=1):
    # stien må innehalda posisjon, fart og tid.

    fps_inv = 1/fps
    # sti = []
    sti_dict = {}

    # sti_komplett = []
    # print(type(tre))
    # tre = ray.get(tre)
    
    solver_args = dict(atol = particle.atol, rtol= particle.rtol, method=particle.method, args = (particle, tre, ribs, L, U), events = (event_check,wrap_check,still_check)        )
 
    step_old = np.concatenate(([particle.init_time], particle.init_position))
    # Step_old og step_new er ein array med [t, x, y, u, v]. 
    
    # sti.append(step_old)
    sti_dict[round(particle.init_time*100)] = dict(position = particle.init_position, loops = 0, caught = False)
    sti_dict['init_time'] = particle.init_time
    
    final_time = particle.init_time

    t = particle.init_time
    t_max = t_span[1]
    t_main = t
    dt_main = fps_inv
    dt = dt_main
    nfev = 0
    
    left_edge = ribs[0].get_rib_middle()[0]
    
    starttid = datetime.datetime.now()

    while True:
        # style = random.randint(0,4)
        text_color = random.randint(30,38)+ 60*random.randint(0,1)
        background = random.randint(40,48) 
        combined = (text_color, background)
        bad = [(30,40),(30,48),(31,41), (32,42), (33,43), (34,44),(35,44),(35,45),(36,46),(37,47),(35,41),(36,42),(37,43),(31,45),(32,46),(33,47),(90,44),(92,42),(93,43),(93,47),(93,47),(96,41),(96,42),(96,45),(97,43),(97,47),(98,43)]
        if (combined not in bad):
            break
    del bad, combined
    status_col = f"{str()};{str(text_color)};{str(background)}"
    
    des4 = ">6.2f"

    status_msg = f"Nr {particle.index}, {particle.diameter:.2f} mm startpos. [{particle.init_position[0]:{des4}},{particle.init_position[1]:{des4}}]  byrja på  t={particle.init_time:.4f}, pos=[{particle.init_position[0]:{des4}},{particle.init_position[1]:{des4}}] U=[{particle.init_position[2]:{des4}},{particle.init_position[3]:{des4}}]"
    print(f"\x1b[{status_col}m {status_msg} \x1b[0m")

    while (t < t_max-2):
        
        particle.collision = check_all_collisions(particle, step_old[1:], ribs)
        if particle.collision['is_resting_contact']:
            particle.resting = True

        try:
            step_new, backcatalog, event, nfev_ny = rk_3(f, (t,t_max), step_old[1:], solver_args, fps)
        except Exception as e:
            raise Exception(f"Feil, med partikkel {particle.index} og tida t0 {t}").with_traceback(e.__traceback__)

        nfev += nfev_ny
        # sti = sti + backcatalog

        for index, step in enumerate(backcatalog):
            sti_dict[round(step[0]*100)] = dict(position = step[1:], loops = particle.wrap_counter, caught = True if step[2] < ribs[1].get_rib_middle()[1] else False)
            final_time = step[0]
            if np.all(index+1 < len(backcatalog) and backcatalog[index+1:,3:] == 0) and event == 'finish': 
                break

        if verbose:
            if (event != "finish"):
                backcatalog = backcatalog + [step_new]
            for step in backcatalog:
                status_msg = f"Nr {particle.index}, {particle.diameter:.2f} mm startpos. [{particle.init_position[0]:{des4}},{particle.init_position[1]:{des4}}] ferdig med t={step[0]:.4f}, pos=[{step[1]:{des4}},{step[2]:{des4}}] U=[{step[3]:{des4}},{step[4]:{des4}}]"
                print(f"\x1b[{status_col}m {status_msg} \x1b[0m")


        if (event== "collision"):
            # collision_info = check_all_collisions(particle, step_new[1:], ribs)
            collision_info = particle.collision
               
            if not collision_correction:
                break

            # assert collision_info['collision_depth'] < eps, collision_info['collision_depth']
            if (collision_info['is_collision']):
                if verbose:
                    print("kolliderte")
                rest = collision_restitution
                particle.resting = False
                    
            elif (collision_info['is_resting_contact']):# and np.dot(collision_info['rib_normal'],collision_info['closest_rib_normal']) == 1.0 ):
                if verbose:
                    print("kvilekontakt")
                if collision_info['rib'].mu == 1:
                    break
                rest = 0
                particle.resting = True

            elif collision_info['is_leaving']:
                if verbose:
                    print("Forlet overflata")
                particle.resting = False
            else:
                app_log.warning(f"Noko feil i kollisjonsinfo for partikkel nr. {particle.index} etter berekninga med t0 {t} og sluttid {final_time}. Det kan sjå ut til at event er collision men det vart ikkje registrert nokon kollisjon.")
                break

            #Gjer alt som skal til for å endra retningen og posisjonen på partikkelen
            step_old = np.copy(step_new)
            
            # assert 'relative_velocity' in collision_info, f"Noko feil i kollisjonsinfo for partikkel nr. {particle.index} etter berekninga med t0 {t} og sluttid {final_time}"
            if 'relative_velocity' not in collision_info:
                app_log.warning(f"Noko feil i kollisjonsinfo for partikkel nr. {particle.index} etter berekninga med t0 {t} og sluttid {final_time}")
                break
            
            n = collision_info['rib_normal']
            v = step_new[3:]
            v_rel = collision_info['relative_velocity'] # v_rel er relativ fart i normalkomponentretning, jf. formel 8-3 i baraff ("notesg.pdf")
            v_new = v - (rest + 1) * v_rel * n

 
            
            step_old[3:] = v_new
            step_old[1:3] = step_old[1:3] + collision_info['rib_normal'] * ε * 0.5
            
        elif (event == "edge"):
            if (particle.wrap_counter <= wrap_max):
                step_old = np.copy(step_new)
                step_old[1] = left_edge
                edgecollision = check_all_collisions(particle, step_old[1:], ribs)
                if edgecollision['is_collision'] or edgecollision['is_resting_contact'] or edgecollision['is_leaving']:
                    step_old[1:3] = step_old[1:3] + edgecollision['rib_normal']*edgecollision['collision_depth']

                particle.wrap_counter += 1
            else:
                break
        elif event == "still":
            if hypot(step_new[3],step_new[4]) < particle.resting_tolerance:
                step_old = np.copy(step_new)
                step_old[3:] = np.zeros(2)
                particle.still = True

        elif (event == "finish"):
            break

        t = step_old[0]

    sti_dict['final_time'] = final_time
    sti_dict['flow_length'] =  ribs[1].get_rib_middle()[0] - left_edge
    
    status_msg = f"Nr. {particle.index} brukte {datetime.datetime.now()-starttid} og kalla funksjonen {nfev} gonger."
    print(f"\x1b[{status_col}m {status_msg} \x1b[0m")    
    # return np.array(sti), sti_dict
    return sti_dict


def rk_3 (f, t, y0, solver_args, fps):
    assert t[1] - t[0] > 0
    solver_args['t_eval'] = eval_steps(t, fps)
    resultat = solve_ivp(f, t, y0, dense_output=True,   **solver_args) # t_eval = [t[1]],
    # har teke ut max_ste=0.02, for det vart aldri aktuelt, ser det ut til.  method=solver_args['method'], args=solver_args['args'],
    # assert resultat.success == True

    if (resultat.message == "A termination event occurred."):
        if resultat.t_events[0].size > 0:
            return np.concatenate((resultat.t_events[0], resultat.y_events[0][0])), np.column_stack((np.asarray(resultat.t), np.asarray(resultat.y).T)), "collision", resultat.nfev
        elif resultat.t_events[1].size > 0:
            return np.concatenate((resultat.t_events[1], resultat.y_events[1][0])), np.column_stack((np.asarray(resultat.t), np.asarray(resultat.y).T)), "edge", resultat.nfev
        elif resultat.t_events[2].size > 0:
            return np.concatenate((resultat.t_events[2], resultat.y_events[2][0])), np.column_stack((np.asarray(resultat.t), np.asarray(resultat.y).T)), "still", resultat.nfev

    else:
        return [], np.column_stack((resultat.t, np.asarray(resultat.y).T)), "finish", resultat.nfev #np.concatenate(([resultat.t[-1]], resultat.y[:,-1]))

def eval_steps(t_span, fps):
    if floor(t_span[0] * 1000000) % floor((1/fps) * 1000000) == 0:
        t_min = ceil(round(t_span[0]+1/fps,5)*fps)/fps
    else:
        t_min = ceil(t_span[0]*fps)/fps
    return np.linspace( t_min, t_span[1], num = round((t_span[1]-t_min)*fps), endpoint = False )

def event_check(t, x, particle, tre, ribs, L, U):
    event_check.counter += 1
    

    collision = check_all_collisions(particle, x, ribs)
    particle.collision = collision
    # collision_depth er negativ når det ikkje er kollisjon og positiv når det er kollisjon.

    # if collision['is_collision'] and collision['inside']:
    #     return collision['collision_depth']
    if collision['is_collision'] or (collision['is_resting_contact'] and particle.resting == False):
        return collision['collision_depth'] - ε #0.0
    elif collision['is_leaving']: # Forlet resting contact og kjem i fri flyt igjen.
        return 0.0

    if collision['is_resting_contact']:
        return -1.0
    return collision['collision_depth'] - ε #skulle kanskje vore berre sett til -1.0?

event_check.counter = 0
event_check.terminal = True

def wrap_check(t, x, particle, tre, ribs, L, U):
    right_edge = ribs[1].get_rib_middle()[0]
        #.strftime('%X.%f')
    if (x[0] > right_edge):
        return 0.0
    return 1.0
wrap_check.terminal = True

def still_check(t,x, particle, tre,ribs, L, U):
    if hypot(x[2],x[3]) < particle.resting_tolerance and hypot(x[2],x[3]) > 0 and particle.resting and not particle.still:
        return 0.0
    # Kvifor har eg denne her? Det er for å dempa farten om den er så bitteliten at han må leggjast til ro. Men det må jo skje berre dersom det er kontakt i tillegg. Så då må eg vel sjekka kollisjon uansett? 
    # Brukte particle.rtol *1 eller particle.rtol * 10, men det verkar til å vera feil uansett. Prøver med 0.01. (OHH 13.12.2021)
    if hypot(x[2],x[3]) > particle.resting_tolerance and particle.still:
        particle.still = False
    return 1.0
still_check.terminal = True