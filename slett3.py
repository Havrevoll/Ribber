from pathlib import Path
from datagenerering import tre_objekt
import pickle

for p in Path("../").glob("*.pickle"):
    h = p.with_suffix(".hdf5")
    nyfil = Path("../ny_pickle").joinpath(p.name)
    if not nyfil.exists():
        tre = tre_objekt(p,h,(0,178))
        print(f"{nyfil} finst ikkje, lagar...")
        with open(nyfil, 'wb') as f:
            pickle.dump(tre, f)
    else:
        print(f"{nyfil} finst alt.")