from datagenerering import tre_objekt
from hjelpefunksjonar import finn_fil
from lag_video import sti_animasjon
from sti_gen import Rib, lag_sti
from sti_gen import Particle


tre_fil = finn_fil(["../Q40_0_60.pickle", "../Q40_0_10.pickle"])
t_span = (0,59)

tre = tre_objekt(tre_fil, t_span)

ribs = [Rib(rib) for rib in tre.ribs]

pa = Particle(0.06, [38.7,-4.5,0,0])

sti = lag_sti(ribs, t_span, pa, tre)
pa.sti = sti
sti_animasjon(pa,(0,10),utfilnamn="ein.mp4")