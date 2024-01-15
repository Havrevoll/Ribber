# -*- coding: utf-8 -*-
'''køyr funksjonar som plottingar(fil['vassføringar'])'''

from operator import truediv
import h5py
import matplotlib

matplotlib.use("Agg")
import datetime
import logging
import random
from math import ceil, floor, hypot, isclose

import numpy as np
import ray
from loguru import logger
from scipy.integrate import solve_ivp

from check_collision import check_all_collisions
from constants import ε
from f import f
from hjelpefunksjonar import f2t, status_colors, t2f

app_log = logging.getLogger(__name__)

from constants import collision_restitution


@ray.remote(max_retries=0)
def remote_lag_sti(f_span, particle, tre, get_u, skalering=1, verbose=True, collision_correction=True):
    return lag_sti(f_span, particle, tre, get_u, skalering=skalering, verbose=verbose, collision_correction=collision_correction)

def lag_sti(f_span, particle, tre, get_u, skalering=1, verbose=True, collision_correction=True):
    # stien må innehalda posisjon, fart og tid.

    # fps_inv = 1/fps
    # sti = []
    sti_dict = {}

    # sti_komplett = []
    # print(type(tre))
    # tre = ray.get(tre)
    
    solver_args = dict(atol = particle.atol, rtol= particle.rtol, method=particle.method, args = (particle, tre, skalering, get_u), events = (event_check,still_check,end_check))
                                                                                                                                                    
 
    step_old = np.concatenate(([particle.init_time], particle.init_position))
    # Step_old og step_new er ein array med [t, x, y, u, v]. 
    
    # sti.append(step_old)
    sti_dict[particle.init_time] = dict(position = particle.init_position, loops = 0, caught = False, time=particle.init_time)
    sti_dict['init_time'] = particle.init_time
    
    final_time = particle.init_time

    frame = particle.init_time
    frame_max = f_span[1]
    # t_main = t
    # dt_main = fps_inv
    # dt = dt_main
    nfev = 0
    
    left_edge = 19.23734
    
    starttid = datetime.datetime.now()

    status_col = status_colors()
    
    des4 = ">6.2f"

    status_msg = f"Nr {particle.index}, {particle.diameter:.2f} mm x₀=[{particle.init_position[0]:{des4}}, {particle.init_position[1]:{des4}}], f₀={particle.init_time} ⇒ t₀={f2t(particle.init_time,skalering):.3f} med {particle.method}"
    print(f"\x1b[{status_col}m {status_msg} \x1b[0m")

    while (frame < frame_max and not isclose(frame,frame_max)):
        # ray.util.pdb.set_trace() 
        particle.collision = check_all_collisions(particle, step_old[1:], ribs)
        if particle.collision['is_resting_contact']:
            particle.resting = True

        try:
            step_new, backcatalog, event, nfev_ny = rk_3(f, (frame,frame_max), step_old[1:], solver_args, skalering=skalering)
        except Exception as e:
            raise Exception(f"Feil, med partikkel {particle.index}, diameter {particle.diameter} og tida f0 {frame}").with_traceback(e.__traceback__)

        nfev += nfev_ny
        # sti = sti + backcatalog

        for index, step in enumerate(backcatalog):
            step[0] = t2f(step[0],skalering)
            sti_dict[round(step[0])] = dict(position = step[1:], loops = particle.wrap_counter, caught = True if (step[2] < ribs[1].get_rib_middle()[1]) or (np.sqrt(np.square(step[3:]).sum()) < 0.1) else False, time=f2t(step[0],skalering))
            final_time = round(step[0])
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
                if collision_info['rib'].µ == 1.5:
                    break
                rest = 0
                particle.resting = True

            elif collision_info['is_leaving']:
                if verbose:
                    print("Forlet overflata")
                particle.resting = False
            else:
                app_log.warning(f"Noko feil i kollisjonsinfo for partikkel nr. {particle.index} etter berekninga med f₀={frame} og sluttid {final_time}. Det kan sjå ut til at event er collision men det vart ikkje registrert nokon kollisjon.")
                break

            #Gjer alt som skal til for å endra retningen og posisjonen på partikkelen
            step_old = np.copy(step_new)
            
            # assert 'relative_velocity' in collision_info, f"Noko feil i kollisjonsinfo for partikkel nr. {particle.index} etter berekninga med t0 {t} og sluttid {final_time}"
            if 'relative_velocity' not in collision_info:
                app_log.warning(f"Noko feil i kollisjonsinfo for partikkel nr. {particle.index} etter berekninga med f₀={frame} og sluttid {final_time}")
                break
            
            n = collision_info['rib_normal'][:,0]
            v = step_new[3:]
            v_rel = collision_info['relative_velocity'] # v_rel er relativ fart i normalkomponentretning, jf. formel 8-3 i baraff ("notesg.pdf")
            v_new = v - (rest + 1) * v_rel * n

 
            
            step_old[3:] = v_new
            step_old[1:3] = step_old[1:3] + collision_info['rib_normal'][:,0] * ε * 0.5
            
        # elif (event == "edge"):
        #     if (particle.wrap_counter <= wrap_max):
        #         step_old = np.copy(step_new)
        #         step_old[1] = left_edge
        #         edgecollision = check_all_collisions(particle, step_old[1:], ribs)
        #         if edgecollision['is_collision'] or edgecollision['is_resting_contact'] or edgecollision['is_leaving']:
        #             step_old[1:3] = step_old[1:3] + edgecollision['rib_normal'][:,0]*edgecollision['collision_depth']

        #         particle.wrap_counter += 1
        #     else:
        #         break
        elif event == "still":
            if hypot(step_new[3],step_new[4]) < particle.resting_tolerance:
                step_old = np.copy(step_new)
                step_old[3:] = np.zeros(2)
                particle.still = True

        elif (event == "finish"):
            break

        frame = t2f(step_old[0],skalering)

    sti_dict['final_time'] = final_time
    # sti_dict['flow_length'] =  ribs[1].get_rib_middle()[0] - left_edge
    
    status_msg = f"Nr. {particle.index} brukte {datetime.datetime.now()-starttid} og kalla funksjonen {nfev} gonger."
    sti_dict['time_usage'] = datetime.datetime.now()-starttid
    print(f"\x1b[{status_col}m {status_msg} \x1b[0m")    
    # return np.array(sti), sti_dict
    return sti_dict


def rk_3 (f, t, y0, solver_args, skalering):
    assert t[1] > t[0]
    
    solver_args['t_eval'] = eval_steps(t, skalering)
    resultat = solve_ivp(f, (f2t(t[0], skalering), f2t(t[1],skalering)), y0, dense_output=True, vectorized=True, **solver_args)
    # t_eval = [t[1]],
    # har teke ut max_ste=0.02, for det vart aldri aktuelt, ser det ut til.  method=solver_args['method'], args=solver_args['args'],
    # assert resultat.success == True

    if (resultat.message == "A termination event occurred."):
        if resultat.t_events[0].size > 0:
            return np.concatenate((resultat.t_events[0], resultat.y_events[0][0])), np.column_stack((np.asarray(resultat.t), np.asarray(resultat.y).T)), "collision", resultat.nfev
        elif resultat.t_events[1].size > 0:
            return np.concatenate((resultat.t_events[1], resultat.y_events[1][0])), np.column_stack((np.asarray(resultat.t), np.asarray(resultat.y).T)), "still", resultat.nfev
        elif resultat.t_events[2].size > 0:
            return np.concatenate((resultat.t_events[2], resultat.y_events[2][0])), np.column_stack((np.asarray(resultat.t), np.asarray(resultat.y).T)), "finish", resultat.nfev 

    else:
        return [], np.column_stack((resultat.t, np.asarray(resultat.y).T)), "finish", resultat.nfev #np.concatenate(([resultat.t[-1]], resultat.y[:,-1]))

def eval_steps(t_span, skalering):
    if isclose(t_span[0],round(t_span[0])):
        return np.asarray([f2t(i,skalering) for i in range(ceil(t_span[0]),t_span[1])])[1:]
    else:
        return np.asarray([f2t(i,skalering) for i in range(ceil(t_span[0]),t_span[1])])

    # if floor(t_span[0] * 1000000) % floor((1/fps) * 1000000) == 0:
    #     t_min = ceil(round(t_span[0]+1/fps,5)*fps)/fps
    # else:
    #     t_min = ceil(t_span[0]*fps)/fps
    # return np.linspace( t_min, t_span[1], num = round((t_span[1]-t_min)*fps), endpoint = False )

def event_check(t, x, particle, tre, ribs, get_u, skalering):
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
    if collision['is_resting_contact'] and collision['rib'] == ribs[2]:
        return 0.0
    if collision['is_resting_contact']:
        return -1.0
    return collision['collision_depth'] - ε #skulle kanskje vore berre sett til -1.0?

event_check.counter = 0
event_check.terminal = True

# def wrap_check(t, x, particle, tre, ribs, get_u, skalering):
#     right_edge = ribs[1].get_rib_middle()[0]
#         #.strftime('%X.%f')
#     if (x[0] > right_edge):
#         return 0.0
#     return 1.0
# wrap_check.terminal = True

def still_check(t,x, particle, tre,ribs, get_u, skalering):
    fart = np.hypot(x[2],x[3])
    if fart < particle.resting_tolerance and fart > 0 and particle.resting and not particle.still:
        return 0.0
    # Kvifor har eg denne her? Det er for å dempa farten om den er så bitteliten at han må leggjast til ro. Men det må jo skje berre dersom det er kontakt i tillegg. Så då må eg vel sjekka kollisjon uansett? 
    # Brukte particle.rtol *1 eller particle.rtol * 10, men det verkar til å vera feil uansett. Prøver med 0.01. (OHH 13.12.2021)
    if fart > particle.resting_tolerance and particle.still:
        particle.still = False
    return 1.0
still_check.terminal = True

def end_check(t,x, particle, tre,ribs, get_u, skalering):
    return particle.length - x[0]
end_check.terminal = True
