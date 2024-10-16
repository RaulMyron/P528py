from p528numpy import P528
import numpy as np

d__km = np.array([0, 200, 300, 400])
h_1__meter = np.array([2, 20, 30, 40])
h_2__meter = np.array([3, 200, 300, 20000])
f__mhz = np.array([1000, 2000, 3000, 4000])
T_pol = np.array([0, 1, 0, 1])  # 0 for horizontal, 1 for vertical
p = np.array([50, 60, 70, 80])

# Call the P528 function with these matrices
result = P528(d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p)