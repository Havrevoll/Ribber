import ray
from ray.exceptions import GetTimeoutError
import numpy as np
import time
ray.init(num_cpus=2)

def a(t):
    time.sleep(t)
    print("sov",t)
    return t

@ray.remote(num_cpus=1)
class Function:
    def __init__(self):
        self._start = None
        
    def execute(self, fn,i):
        self._start = time.time()
        return fn(i)
        
    def duration(self):
        return time.time() - self._start
        
funcs = [Function.remote() for i in range(4)] # the actors

b = [2,3,4,5000,6000,7000,8000,9000,10000,11000,12000,13,14,15,16]
b = [2,3000,4000,5]
everyone = {f:{'ex':f.execute.remote(a,i)} for i,f in zip(b,funcs)}

del funcs


all_durations = {f.duration.remote():f for f in everyone}
unfinished = list(all_durations.keys())
while True:
    
    dur_result,unfinished = ray.wait(list(all_durations.keys()))
    duration = ray.get(dur_result)
    if duration[0] > 10:
        try:
            ray.get(everyone[all_durations[dur_result[0]]]['ex'],timeout=1)
            ray.kill(all_durations[dur_result[0]])
            del everyone[all_durations[dur_result[0]]]
            all_durations = {f.duration.remote():f for f in everyone}
            # del all_durations[dur_result[0]]
        except GetTimeoutError:
            unfinished.append(dur_result[0])
    else:
        f = all_durations.pop(dur_result[0])
        all_durations[f.duration.remote()] = f
        unfinished = list(all_durations.keys())
    if len(dur_result) == 0:
        break
    # fetch result with ray.get
    # if result > 10, kill the actor (use a dictionary to map result to actor)
    # otherwise, call actor.duration.remote() again and add the future back into all_duration.