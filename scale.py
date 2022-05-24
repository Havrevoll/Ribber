import numpy as np
import h5py


def scale(infile, outfile,factor):
    from math import sqrt

    with h5py.File(infile,'r') as f:
        u = np.asarray(f['Umx'])*sqrt(factor)
        v = np.asarray(f['Vmx'])*sqrt(factor)
        x = np.asarray(f['x'])*factor
        y = np.asarray(f['y'])*factor
        ribs = np.asarray(f['ribs'])*factor
        I,J = f.attrs['I'], f.attrs['J']
        A = f.attrs['A']
        L = f.attrs['L']
        Q = f.attrs['Q']
        U = f.attrs['U']  
    with h5py.File(outfile, 'w') as f:
        f.create_dataset("x", data=x, compression="gzip", compression_opts=9)
        f.create_dataset("y", data=y, compression="gzip", compression_opts=9)
        f.create_dataset("Umx", data=u, compression="gzip", compression_opts=9)
        f.create_dataset("Vmx", data=v, compression="gzip", compression_opts=9)
        f.attrs['A'] = A
        f.attrs['I'], f.attrs['J'] = I,J
        f.attrs['L'] = L
        f.attrs['Q'] = Q
        f.attrs['U'] = U
        f.create_dataset('ribs', data=ribs)