import numpy as np
from pathlib import Path
import h5py
import pickle
import csv

# def data_import(path='C:/Users/havrevol/PIV/Three/Q20'):

p = Path('C:/Users/havrevol/PIV/Three/Q20/Analysis')


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
    
    for q in p.glob('./*.vec'):
    
        u_single = []
        v_single = []    
    
        with q.open() as f:
            csvreader = csv.reader(f)
            rows = [fields for fields in csvreader]
    
        exec(rows[0][6].strip())
        exec(rows[0][7].strip())
     
        for row in rows[1:]:
            u_single.append(float(row[2]))
            v_single.append(float(row[3]))
            
        u.append(u_single)
        v.append(v_single)
        
    u = np.array(u)
    v = np.array(v)
    x = np.array(x)
    y = np.array(y)
    
    return x,y,u,v

x,y,u,v = import_data(p)

def save_to_file(path='C:/Users/havrevol/PIV/Three/Q20/Q20.hdf5'):
     with h5py.File(path, 'w') as f:
         f.create_dataset("x", data=x, compression="gzip", compression_opts=9)
         f.create_dataset("y", data=y, compression="gzip", compression_opts=9)
         f.create_dataset("Umx", data=u, compression="gzip", compression_opts=9)
         f.create_dataset("Vmx", data=v, compression="gzip", compression_opts=9)
         
save_to_file()
         
    
#%%
    
'''
files = dir( fullfile(pwd,'*.vec') ); % Get all the files with extension
                                      % vec in the directory

for sz = 1:3600     % define how many data files
    disp(sz);
    filename = files(sz,1).name;
    disp(filename);
    fid= fopen(filename,'r');
    Fl = fgetl(fid);
    
    ftype1 = strfind(Fl,'Standard Uncertainty');
    ftype2 = strfind(Fl,'Expanded Uncertainty');
    if ~isempty(ftype1) && ~isempty(ftype2)
        VEL = textscan(fid,'%f, %f, %f, %f, %d, %f, %f, %f, %f');
        x   = VEL{1};
        y   = VEL{2};
        u   = VEL{3};
        v   = VEL{4};
        CHC = VEL{5};
        LBSTUC = VEL{6}./sqrt(2);
        UBSTUC = VEL{7}./sqrt(2);
        LBEXUC = VEL{8}./sqrt(2);
        UBEXUC = VEL{9}./sqrt(2);
    else
        VEL = textscan(fid,'%f, %f, %f, %f, %d');
        x   = VEL{1};
        y   = VEL{2};
        u   = VEL{3};
        v   = VEL{4};
        CHC = VEL{5};
    end
    
    %Conditions
    con_1   = CHC < 0;
    con_2   = u < -2;
    con_3   = u > 2;
    con_4   = v < -2;
    con_5   = v > 2;
   
    u(con_1)= nan;
    v(con_1)= nan;
    u(con_2)= nan;
    v(con_2)= nan;
    u(con_3)= nan;
    v(con_3)= nan;
    u(con_4)= nan;
    v(con_4)= nan;
    u(con_5)= nan;
    v(con_5)= nan;
    fclose(fid);
    U{sz,1}(:,1) = u;
    V{sz,1}(:,1) = v; 
    LSUC{sz,1}(:,1) = LBSTUC;
    USUC{sz,1}(:,1) = UBSTUC;
    LEUC{sz,1}(:,1) = LBEXUC;
    UEUC{sz,1}(:,1) = UBEXUC;
end
Umx = cat(3,[U{:,1}]);
Vmx = cat(3,[V{:,1}]);
LSUCmx = cat(3,[LSUC{:,1}]);
USUCmx = cat(3,[USUC{:,1}]);
LEUCmx = cat(3,[LEUC{:,1}]);
UEUCmx = cat(3,[UEUC{:,1}]);
save('Umx3600.mat','Umx','-v7.3');
save('Vmx3600.mat','Vmx','-v7.3');
save('X3600.mat','x','-v7.3');
save('Y3600.mat','y','-v7.3');
save('LSUCmx3600.mat','LSUC','-v7.3');
save('USUCmx3600.mat','USUC','-v7.3');
save('LEUCmx3600.mat','LEUC','-v7.3');
save('UEUCmx3600.mat','UEUC','-v7.3');
%% Load Data
% Check the file names before loading

'''