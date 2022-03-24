import numpy as np
import h5py
from pathlib import Path
import pickle
import re
from math import pi as π



namneliste = ["rib25_Q20_1",
"rib25_Q20_2",
"rib25_Q20_3",
"rib25_Q40_1",
"rib25_Q40_2",
"rib25_Q60_1",
"rib25_Q60_2",
"rib25_Q80_1",
"rib25_Q80_2",
"rib25_Q100_1",
"rib25_Q100_2",
"rib75_Q20_1",
"rib75_Q40_1",
"rib75_Q40_2",
"rib75_Q40_3",
"rib75_Q60_1",
"rib75_Q80_1",
"rib75_Q80_2",
"rib75_Q80_3",
"rib75_Q100_1",
"rib75_Q100_2",
"rib75_Q100_3",
"rib75_Q100_4",
"rib50_Q20_1",
"rib50_Q20_2",
"rib50_Q20_3",
"rib50_Q40_1",
"rib50_Q60_1",
"rib50_Q80_1",
"rib50_Q100_1",
"rib50_Q120_1",
"rib50_Q140_1"]

for namn in namneliste:
    namn = Path("data").joinpath(namn).with_suffix(".hdf5")
    delar = re.split('_', namn.stem)

    L = int(re.search('[0-9]+', delar[0])[0])
    Q = int(re.search('[0-9]+', delar[1])[0])
    forsøk = int(delar[2])

    A = (5.5**2 * π / 2 + (12 - 5.5) * 11) / 20**2

    U = Q/A

    with h5py.File(namn,'r+') as f:
        # (I,J) = (int(np.asarray(f['I'])),int(np.asarray(f['J'])))
        # f.attrs['I'] = I
        # f.attrs['J'] = J
        # f.attrs['L'] = L
        # f.attrs['Q'] = Q
        # f.attrs['A'] = A
        f.attrs['U'] = U


