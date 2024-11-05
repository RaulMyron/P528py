#flake8: noqa

import os
import re
import numpy as np
from glob import glob
from p528 import *
import time

paths = ['../Data Tables']

terminal_1 = Terminal()
terminal_2 = Terminal()
tropo = TroposcatterParams()
path = Path()
los_params = LineOfSightParams()
result = Result()

cnt_fail = 0
cnt_pass = 0

start_time = time.time()

for path_index in paths:
    filenames = glob(os.path.join(path_index, '*.csv'))
    datanumber = len(filenames)
    
    for ii, filename in enumerate(filenames, 1):
        print('*' * 47)
        print(f' Processing file {ii}/{datanumber}: {filename} ...')
        print('*' * 47)
        
        with open(filename, 'r') as fid:
            # First line processing
            readLine = fid.readline().strip()
            rgx = r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?'
            dummy = re.findall(rgx, readLine)
            f = float(dummy[0])
            p = float(dummy[1])
            
            # h2 values
            readLine = fid.readline().strip()
            dummy = readLine.split(',')
            h2 = [float(x) for x in dummy[2:-3]]
            
            # h1 values
            readLine = fid.readline().strip()
            dummy = readLine.split(',')
            h1 = [float(x) for x in dummy[2:-3]]
            
            fid.readline()  # Skip a line
            
            print(f'{"PYTHON":>20} {"REF TABLE":>20} {"DELTA":>20}')
            
            D, FSL, tl_ref = [], [], []
            for line in fid:
                dummy = line.strip().split(',')
                D.append(float(dummy[0]))
                FSL.append(float(dummy[1]))
                tl_ref.append([float(x) for x in dummy[2:-3]])
            
        D = np.array(D)
        tl_ref = np.array(tl_ref)
        
        for i in range(0, len(D), 1000):
            for j in range(len(h1)):

                result = P528_Ex(D[i], h1[j], h2[j], f, 0, p*100, result,
                                        terminal_1, terminal_2, tropo, path, los_params)

                
                delta = round(10.0 * (result.A__db - tl_ref[i,j])) / 10.0
                
                if abs(delta) > 0.1:
                    cnt_fail += 1
                else:
                    cnt_pass += 1
                
                print(f'{result.A__db:20.1f} {tl_ref[i,j]:20.1f} {delta:20.1f}')
                
                result.clear(); terminal_1.clear(); terminal_2.clear(); tropo.clear(); path.clear(); los_params.clear();

end_time = time.time()
elapsed_time = end_time - start_time

print(f'Successfully passed {cnt_pass} out of {cnt_pass+cnt_fail} tests')
print(f'Time taken: {elapsed_time:.2f} seconds')