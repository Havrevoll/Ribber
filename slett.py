# from numpy import pi
# from datagenerering import tre_objekt
# from hjelpefunksjonar import finn_fil
from lag_video import sti_animasjon
from sti_gen import Rib, lag_sti
from sti_gen import Particle
from pathlib import Path
from sti_gen import f
import pickle

t_span = (0,179)

pickle_fil = Path("../TONSTAD_TWO_Q20_TWO2.pickle")
hdf5_fil = pickle_fil.with_suffix(".hdf5")

with open(pickle_fil,'rb') as f:
    tre = pickle.load(f)


ribs = [Rib(rib) for rib in tre.ribs]

# pa = Particle(0.06, [38.7,-4.5,0,0])
pa = Particle(0.06357829445507623, [-35.81, 12.09278197011611, 0,0])
pa.init_time = 42.35
pa2 = Particle(0.10347144483338566, [-35.81, 68.73971570789526,0,0])
pa2.init_time = 12.75
# pa3 = Particle(0.3, [-8,10.0,0,0])
# pa4 = Particle(0.1, [-70,70.0,0,0])
# pa5 = Particle(0.1, [-70,70.0,0,0])
pa.atol = pa.rtol =pa2.atol = pa2.rtol = 1e-1
pa.method =pa2.method = 'RK23'
# pa2.atol = pa3.atol = pa2.rtol = pa3.rtol = 1e-1
pa2.index = 1
# pa3.index = 2
# pa4.index = 3
# pa5.index = 4

pa.sti_dict = lag_sti(ribs, t_span, pa, tre, verbose=False)
pa2.sti_dict = lag_sti(ribs, t_span, pa2, tre, verbose=False)
# pa3.sti_dict = lag_sti(ribs, t_span, pa3, tre, verbose=False)
# pa4.sti_dict = lag_sti(ribs, t_span, pa4, tre, verbose=False)
# pa5.sti_dict = lag_sti(ribs, t_span, pa5, tre, verbose=False)
sti_animasjon([pa, pa2], ribs, t_span,hdf5_fil, utfilnamn="ein_utan_blit.mp4")

from subprocess import run
run("rsync ein_utan_blit.mp4 havrevol@login.ansatt.ntnu.no:", shell=True)