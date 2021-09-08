import numpy as np
from pathlib import Path
import h5py
import csv

# def data_import(path='C:/Users/havrevol/PIV/Three/Q20'):

# p = Path('C:/Users/havrevol/PIV/Three/Q20/Analysis')
mappa = input("Kor er mappa med *.vec-filene?")
p= Path(mappa)
save = input("Kor skal hdf5-fila lagrast?")

def import_data(p):
    x = []
    y = []
    
    u = []
    v = []
    
    for q in p.glob('./*.vec'):
        with q.open() as f:
            csvreader = csv.reader(f)
            rows = [fields for fields in csvreader]
    
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
            
        # exec(rows[0][6].strip())
        # exec(rows[0][7].strip())
     
        for row in rows[1:]:
            if (-2 < int(row[4]) < 2): # Check the CHC
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
    I=126
    J=127
    
    return x,y,u,v,I,J

def save_to_file(path=save):
     with h5py.File(path, 'w') as f:
         f.create_dataset("x", data=x, compression="gzip", compression_opts=9)
         f.create_dataset("y", data=y, compression="gzip", compression_opts=9)
         f.create_dataset("Umx", data=u, compression="gzip", compression_opts=9)
         f.create_dataset("Vmx", data=v, compression="gzip", compression_opts=9)
         f.create_dataset('I', data=I)
         f.create_dataset('J', data=J)

x,y,u,v, I,J = import_data(p)
         
save_to_file()
         