# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 09:22:39 2021

@author: havrevol
"""
import matplotlib
matplotlib.use("Agg")
import pickle
from sti_gen import Rib, simulering #,Particle
from lag_video import sti_animasjon
from hjelpefunksjonar import finn_fil
import datetime
from pathlib import Path
import logging
import h5py

tal = 1000
rnd_seed=1
tider = {}

pickle_filer = [#"TONSTAD_FOUR_Q20_FOUR TRIALONE.pickle",
# "TONSTAD_FOUR_Q20_FOUR CHECK.pickle",
# "TONSTAD_FOUR_Q20_FOUR REPEAT.pickle",
# "TONSTAD_FOUR_Q40_FOUR.pickle",
"TONSTAD_FOUR_Q40_REPEAT.pickle",
"TONSTAD_FOUR_Q60_FOUR.pickle",
"TONSTAD_FOUR_Q60_FOUR REPEAT.pickle",
"TONSTAD_FOUR_Q80_FOURDTCHANGED.pickle",
"TONSTAD_FOUR_Q80_FOUR.pickle",
"TONSTAD_FOUR_Q100_FOUR DT.pickle",
"TONSTAD_FOUR_Q100_FOUR.pickle",
"Tonstad_THREE_Q20_THREE.pickle",
"Tonstad_THREE_Q40_THREE.pickle",
"Tonstad_THREE_Q40_THREE_EXTRA.pickle",
"Tonstad_THREE_Q40_THREE FINAL.pickle",
"Tonstad_THREE_Q60_THREE.pickle",
"Tonstad_THREE_Q80_THREE.pickle",
"Tonstad_THREE_Q80_THREE_EXTRA.pickle",
"Tonstad_THREE_Q80EXTRA2_THREE.pickle",
"Tonstad_THREE_Q100_THREE.pickle",
"Tonstad_THREE_Q100_THREE_EXTRA.pickle",
"Tonstad_THREE_Q100_EXTRA2_THREE.pickle",
"Tonstad_THREE_Q100_THREE_EXTRA3.pickle",
"TONSTAD_TWO_Q20_TWO.pickle",
"TONSTAD_TWO_Q20_TWO2.pickle",
"TONSTAD_TWO_Q20_TWO3.pickle",
"TONSTAD_TWO_Q40_TWO.pickle",
"TONSTAD_TWO_Q60_TWO.pickle",
"TONSTAD_TWO_Q80_TWO.pickle",
"TONSTAD_TWO_Q100_TWO.pickle",
"TONSTAD_TWO_Q120_TWO.pickle",
"TONSTAD_TWO_Q140_TWO.pickle"]

# logging.basicConfig(filename='simuleringar.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s')


log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s %(funcName)s(%(lineno)d) %(message)s')

#File to log to
logFile = 'simuleringar.log'

#Setup File handler
file_handler = logging.FileHandler(logFile)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.DEBUG)

#Setup Stream Handler (i.e. console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
stream_handler.setLevel(logging.INFO)

#Get our logger
app_log = logging.getLogger()

#Add both Handlers
app_log.addHandler(file_handler)
app_log.addHandler(stream_handler)

app_log.setLevel(logging.DEBUG)


for pickle_namn in pickle_filer:

    # pickle_namn = "TONSTAD_FOUR_Q40_REPEAT.pickle"
    # assert pickle_fil.exists()
    pickle_fil = finn_fil([Path("..").joinpath(Path(pickle_namn)), Path("~/hard/").joinpath(Path(pickle_namn)).expanduser(), Path("/mnt/g/pickle/").joinpath(Path(pickle_namn))])

    assert pickle_fil.exists() and pickle_fil.with_suffix(".hdf5").exists()

    talstart = datetime.datetime.now()
    app_log.info(f"No er det {pickle_namn}")
    app_log.info("Skal henta tre.")
    with open(pickle_fil,'rb') as f:
        tre = pickle.load(f)
    app_log.info("Ferdig å henta tre.")

    rib_data = tre.ribs

    rib_namn = Path(pickle_fil.parent.joinpath(f"{pickle_fil.stem}_tre.h5py"))
    with h5py.File(rib_namn, 'w') as f:
        f.create_dataset('ribs', data=rib_data)

    # ribs = [Rib(rib) for rib in tre.ribs]
  