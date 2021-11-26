# from numpy import pi
# from datagenerering import tre_objekt
# from hjelpefunksjonar import finn_fil
from lag_video import sti_animasjon
from sti_gen import Rib, lag_sti
from sti_gen import Particle
from pathlib import Path
from sti_gen import f
import pickle

t_span = (0,177)

pickle_fil = Path("../TONSTAD_TWO_Q20_TWO2.pickle")
hdf5_fil = pickle_fil.with_suffix(".hdf5")

with open(pickle_fil,'rb') as f:
    tre = pickle.load(f)


ribs = [Rib(rib) for rib in tre.ribs]

# pa = Particle(0.06, [38.7,-4.5,0,0])
pa = Particle(0.06, [0,0,0,0])
pa2 = Particle(0.06, [0,0,0,0])
pa3 = Particle(0.03, [-70,70.0,0,0])
pa4 = Particle(0.01, [-70,70.0,0,0])
pa4.atol = pa2.atol = pa3.atol = pa4.rtol = pa2.rtol = pa3.rtol = 1e-1
pa2.index = 1
pa3.index = 2
pa4.index = 3

pa.sti = lag_sti(ribs, t_span, pa, tre, verbose=False)
pa2.sti = lag_sti(ribs, t_span, pa2, tre, verbose=False)
pa3.sti = lag_sti(ribs, t_span, pa3, tre, verbose=False)
pa4.sti = lag_sti(ribs, t_span, pa4, tre, verbose=False)
sti_animasjon([pa,pa2,pa3,pa4], ribs, t_span,hdf5_fil, utfilnamn="ein_utan_blit.mp4")