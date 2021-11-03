from numpy import pi
from datagenerering import tre_objekt
# from hjelpefunksjonar import finn_fil
from lag_video import sti_animasjon
from sti_gen import Rib, lag_sti
from sti_gen import Particle
from pathlib import Path


# tre_fil = finn_fil(["../Q40_0_60.pickle", "../Q40_0_10.pickle"])
t_span = (0,178)

pickle_fil = Path("../TONSTAD_TWO_Q20_TWO2.pickle")
hdf5_fil = Path("../TONSTAD_TWO_Q20_TWO2.hdf5")

tre = tre_objekt(pickle_fil, hdf5_fil, t_span)

ribs = [Rib(rib) for rib in tre.ribs]

pa = Particle(0.06, [38.7,-4.5,0,0])

# sti = lag_sti(ribs, t_span, pa, tre)
# pa.sti = sti
# sti_animasjon(pa,(0,10),utfilnamn="ein.mp4")