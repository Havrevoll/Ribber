from numpy import pi
from datagenerering import tre_objekt
# from hjelpefunksjonar import finn_fil
from lag_video import sti_animasjon
from sti_gen import Rib, lag_sti
from sti_gen import Particle
from pathlib import Path
from sti_gen import f
import pickle

t_span = (0,178)

pickle_fil = Path("../TONSTAD_TWO_Q20_TWO2.pickle")
hdf5_fil = pickle_fil.with_suffix(".hdf5")

with open(pickle_fil,'rb') as f:
    tre = pickle.load(f)


ribs = [Rib(rib) for rib in tre.ribs]

# pa = Particle(0.06, [38.7,-4.5,0,0])
pa = Particle(0.06, [11.15,-73.0,26.9,0])

sti = lag_sti(ribs, t_span, pa, tre)
pa.sti = sti
# sti_animasjon([pa],(0,178),hdf5_fil, utfilnamn="ein.mp4")