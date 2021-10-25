import numpy as np
from pathlib import Path
import h5py
import csv
import re
import ray


from numpy.lib.npyio import save

# import argparse
# parser = argparse.ArgumentParser(description="Hent vec-filer og gjer om til hdf5-fil")
# parser.add_argument("vec_folder", help='Mappa med vec-filene i.')
# parser.add_argument("hdf5",  help='hdf5-fila som skal lagrast til')

# args = parser.parse_args()
# p = Path(args.vec_folder)
# save = Path(args.savepath)

# if not p.is_dir():
#     raise Exception("Denne vec-mappa finst ikkje.")

# if not save.parent.resolve().exists():
#     raise Exception("Mappa for hdf5-fila finst visst ikkje.")

maskefiler = {'Q100_FOUR': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q100_FOUR DT': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q100 DTCHANGED': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q20_FOUR TRIALONE': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q20_FOUR CHECK': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q20_FOUR REPEAT': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q20_FOURDTDECREASE': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q40_FOUR': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q40_REPEAT': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q60_FOUR': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q60_FOUR REPEAT': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q80_FOUR': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q80_FOUR TRIAL': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q80_FOURDTCHANGED': {'exp': 'TONSTAD_FOUR', 'rib': 'Mask_211020PtsL.102021152033.PPL'},
 'Q20_THREE': {'exp': 'Tonstad_THREE', 'rib': 'Mask211014PtsL.101421114839.PPL'},
 'Q40_THREE': {'exp': 'Tonstad_THREE', 'rib': 'mask211014_2PtsL.101421114934.PPL'},
 'Q40_THREE_EXTRA': {'exp': 'Tonstad_THREE', 'rib': 'mask211014_2PtsL.101421114934.PPL'},
 'Q40_THREE_FINAL': {'exp': 'Tonstad_THREE', 'rib': 'Mask211014_4PtsL.101421115020.PPL'},
 'Q40_THREE_REPEAT': {'exp': 'Tonstad_THREE', 'rib': 'Mask211014_3PtsL.101421114955.PPL'},
 'Q60_THREE': {'exp': 'Tonstad_THREE', 'rib': 'mask211014_2PtsL.101421114934.PPL'},
 'Q80_THREE': {'exp': 'Tonstad_THREE', 'rib': 'mask211014_2PtsL.101421114934.PPL'},
 'Q80_THREE_EXTRA': {'exp': 'Tonstad_THREE', 'rib': 'Mask211014_3PtsL.101421114955.PPL'},
 'Q80_THREE_EXTRA2': {'exp': 'Tonstad_THREE', 'rib': 'Mask211014_5PtsL.101421115031.PPL'},
 'Q100_THREE': {'exp': 'Tonstad_THREE', 'rib': 'Mask211014_4PtsL.101421115020.PPL'},
 'Q100_THREE_EXTRA': {'exp': 'Tonstad_THREE', 'rib': 'Mask211014_4PtsL.101421115020.PPL'},
 'Q100_THREE_EXTRA2': {'exp': 'Tonstad_THREE', 'rib': 'Mask211014_5PtsL.101421115031.PPL'},
 'Q20_TWO': {'exp': 'TONSTAD_TWO', 'rib': 'two_211001PtsL.100421111059.PPL'},
 'Q40_TWO': {'exp': 'TONSTAD_TWO', 'rib': 'two_211001PtsL.100421111059.PPL'},
 'Q60_TWO': {'exp': 'TONSTAD_TWO', 'rib': 'two_211001PtsL.100421111059.PPL'},
 'Q80_TWO': {'exp': 'TONSTAD_TWO', 'rib': 'two_211001PtsL.100421111059.PPL'},
 'Q100_TWO': {'exp': 'TONSTAD_TWO', 'rib': 'two_211001PtsL.100421111059.PPL'},
 'Q120_TWO': {'exp': 'TONSTAD_TWO', 'rib': 'two_211001PtsL.100421111059.PPL'},
 'Q140_TWO': {'exp': 'TONSTAD_TWO', 'rib': 'two_211001PtsL.100421111059.PPL'}}

@ray.remote
def import_data(p):
    x = []
    y = []
    
    u = {}
    v = {}
    
    run = p.parent.stem

    for q in p.glob('./*.vec'): # Hentar ut I, J, x og y frå berre éi fil.
        with q.open() as f:
            csvreader = csv.reader(f)
            rows = [fields for fields in csvreader]
    
        I = int(re.search(' I=(\d{1,3})',rows[0][10]).group(1))
        J = int(re.search(' J=(\d{1,3})',rows[0][11]).group(1))
        x_resolution = float(re.search('MicrometersPerPixelX="(\d+\.\d+)"',rows[0][9]).group(1))/1000.0
        origo = np.array([float(re.search('OriginInImageX="(\d+\.\d+)"',rows[0][9]).group(1)), float(re.search('OriginInImageY="(\d+\.\d+)"',rows[0][9]).group(1))])
        for row in rows[1:]:
            x.append(float(row[0]))
            y.append(float(row[1]))

        print("Ferdig med å lesa x og y") 
        break
    
    i=0
    
    for q in p.glob('./*.vec'):
    
        
        u_single = []
        v_single = []    
    
        with q.open() as f:
            csvreader = csv.reader(f)
            rows = [fields for fields in csvreader]
 
    
        if (i %100 == 0):
            # print(rows[0][0]) 
            print("Har lese {} filer i {}".format(i, p.name))
            
        for row in rows[1:]:
            if (0 < int(row[4])): # Check the CHC
                u_single.append(float(row[2])*1000.)
                v_single.append(float(row[3])*1000.)
            else:
                u_single.append(np.nan)
                v_single.append(np.nan)
                
        filename = int(re.search("\d\d\d\d\d\d",q.name).group())
        u[filename] = u_single
        v[filename] = v_single
        
        i += 1

        # if (i > 100):
        #     break
        
    u = np.array([u[x] for x  in sorted(u)])
    v = np.array([v[x] for x  in sorted(v)])
    x = np.array(x)
    y = np.array(y)
    
    ribs = []

    with open(p.parents[1].joinpath("Settings").joinpath(maskefiler[run]['rib'])) as f:
        for l in f:
            ribs.append(l.split(","))
    
    ribs = ( (np.array([0,2047]) - np.array(ribs).astype(float) ) * np.array([-1,1]) - origo ) * x_resolution
    
    return x,y,u,v,I,J, ribs

def save_to_file(path, x,y,u,v,I,J,ribs):
    print("Skal lagra i {}".format(path))
    with h5py.File(path, 'w') as f:
        f.create_dataset("x", data=x, compression="gzip", compression_opts=9)
        f.create_dataset("y", data=y, compression="gzip", compression_opts=9)
        f.create_dataset("Umx", data=u, compression="gzip", compression_opts=9)
        f.create_dataset("Vmx", data=v, compression="gzip", compression_opts=9)
        f.create_dataset('I', data=I)
        f.create_dataset('J', data=J)
        f.create_dataset('ribs', data=ribs)
    print("Ferdig å lagra {}".format(path))


# x,y,u,v, I,J, ribs = import_data(p)

# ribs = np.array([[ [-10.83594, -1.1020], [-11.0196, -8.9075], [-60.79146, -0.8265], [-60.97512, -8.6320] ], 
#  [ [89.0751, 0.09183], [89.0751, -7.71372], [39.02775, 0.09183], [39.02775, -7.71372] ],
# [[93.29928, -74.84145], [-93.20745, -72.63753], [-93.20745, -98.80908], [93.29928,- 98.80908]]])
         
# save_to_file()


eks = { "Tonstad_THREE":['Q100_THREE', 'Q100_THREE_EXTRA', 'Q20_THREE', 'Q40_THREE', 'Q40_THREE FINAL', #'Q100_EXTRA2_THREE', 'Q40_THREE REPEAT', 
                'Q40_THREE_EXTRA', 'Q60_THREE', 'Q80EXTRA2_THREE', 'Q80_THREE', 'Q80_THREE_EXTRA'], 
        #"TONSTAD_TWO":['Q100_TWO', 'Q120_TWO', 'Q140_TWO', 'Q20_TWO', 'Q40_TWO', 'Q60_TWO', 'Q80_TWO'], 
    "TONSTAD_FOUR":['Q100 DTCHANGED', 'Q100_FOUR', 'Q100_FOUR DT', 'Q20_FOUR CHECK', 
        'Q20_FOUR REPEAT', 'Q20_FOUR TRIALONE', 
        'Q20_FOURDTDECREASE', 'Q40_FOUR']}#, 'Q40_REPEAT', 'Q60_FOUR', 'Q60_FOUR REPEAT', 'Q80_FOUR', 'Q80_FOURDTCHANGED']}

eks = { "TONSTAD_FOUR":['Q100_FOUR', 'Q100_FOUR DT', 'Q20_FOUR CHECK', 'Q20_FOUR REPEAT', 'Q20_FOUR TRIALONE', 'Q40_FOUR', 'Q40_REPEAT', 'Q60_FOUR', 'Q60_FOUR REPEAT', 'Q80_FOUR','Q80_FOURDTCHANGED']}

# def create_hdf5(eks):
ray.init()

jobs = {}

for e,runs in eks.items():
    for r in runs:
        folder = Path("../vec").joinpath(e).joinpath(r).joinpath("Analysis")
        # folder = Path("/mnt/g/Experiments11/").joinpath(e).joinpath(r).joinpath("Analysis")
        # print(str(folder), len(list(folder.glob("*.TIF")))/2)
        
        if (not folder.exists()):
            raise Exception("{} finst ikkje".format(folder))

        hdf5_file = Path("..").joinpath("{}_{}.hdf5".format(e,r))
        
        jobs[r] ={'id':import_data.remote(folder), 'hdf5':hdf5_file}

    
for k in jobs:
    
    hdf5_file = jobs[k]['hdf5']
    print("hentar jobben til ",k, hdf5_file)
    data = ray.get(jobs[k]['id'])
    save_to_file(hdf5_file, *data)
