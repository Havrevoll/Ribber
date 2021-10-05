import numpy as np
from pathlib import Path
import h5py
import csv
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("vec_folder")
parser.add_argument("savepath")

args = parser.parse_args()
p = Path(args.vec_folder)
save = Path(args.savepath)

if not p.is_dir():
    raise Exception("Denne vec-mappa finst ikkje.")

if not save.parent.resolve().exists():
    raise Exception("Mappa for hdf5-fila finst visst ikkje.")

# def data_import(path='C:/Users/havrevol/PIV/Three/Q20'):

# p = Path('C:/Users/havrevol/PIV/Three/Q20/Analysis')
# while True:
#     mappa = input("Kor er mappa med *.vec-filene?")
#     if os.path.isdir(mappa):
#         break
#     print("Mappa finst ikkje.")

# while True:
#     save = input("Skriv fullstendig bane til hdf5-fila:")
#     sti, fil = os.path.split(save)
#     if os.path.isdir(sti):
#         break
#     print("Mappa finst ikkje.")

def import_data(p):
    x = []
    y = []
    
    u = []
    v = []
    
    for q in p.glob('./*.vec'): # Hentar ut I, J, x og y frå berre éi fil.
        with q.open() as f:
            csvreader = csv.reader(f)
            rows = [fields for fields in csvreader]
    
        I = int(re.search(' I=(\d{1,3})',rows[0][10]).group(1))
        J = int(re.search(' J=(\d{1,3})',rows[0][11]).group(1))
        for row in rows[1:]:
            x.append(float(row[0]))
            y.append(float(row[1]))
        
        break
    
    i=0
    
    for q in p.glob('./*.vec'):
    
        
        u_single = []
        v_single = []    
    
        with q.open() as f:
            csvreader = csv.reader(f)
            rows = [fields for fields in csvreader]
 
    
        if (i %100 == 0):
            print(rows[0][0]) 
            
        for row in rows[1:]:
            if (0 < int(row[4])): # Check the CHC
                u_single.append(float(row[2])*1000)
                v_single.append(float(row[3])*1000)
            else:
                u_single.append(np.nan)
                v_single.append(np.nan)
                
            
        u.append(u_single)
        v.append(v_single)
        
        i += 1
        
    u = np.array(u)
    v = np.array(v)
    x = np.array(x)
    y = np.array(y)
    
    
    return x,y,u,v,I,J

def save_to_file(path=save):
     with h5py.File(path, 'w') as f:
         f.create_dataset("x", data=x, compression="gzip", compression_opts=9)
         f.create_dataset("y", data=y, compression="gzip", compression_opts=9)
         f.create_dataset("Umx", data=u, compression="gzip", compression_opts=9)
         f.create_dataset("Vmx", data=v, compression="gzip", compression_opts=9)
         f.create_dataset('I', data=I)
         f.create_dataset('J', data=J)
         f.create_dataset('ribs', data=ribs)


x,y,u,v, I,J = import_data(p)

ribs = np.array([[ [-10.83594, -1.1020], [-11.0196, -8.9075], [-60.79146, -0.8265], [-60.97512, -8.6320] ], 
 [ [89.0751, 0.09183], [89.0751, -7.71372], [39.02775, 0.09183], [39.02775, -7.71372] ],
[[93.29928, -74.84145], [-93.20745, -72.63753], [-93.20745, -98.80908], [93.29928,- 98.80908]]])
         
save_to_file()
         