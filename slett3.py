from pathlib import Path
from datagenerering import tre_objekt
import pickle

for p in Path("../").glob("*.pickle"):
    # nyfil = Path("../ny_pickle").joinpath(p.name)
    print(f"Skal henta {p.name}.")
    with open(p, 'rb') as f:
        tre = pickle.load(f)
    u = tre.ribs
    tre.ribs = tre.U_kd
    tre.U_kd = u
    with open(p, 'wb') as f:
        pickle.dump(tre, f)
    print(f"ferdig å lagra {p.name}.")
