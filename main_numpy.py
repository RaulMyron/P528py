from p528numpy import P528
import numpy as np

# Assuming the P528 function is imported or defined as discussed earlier

# Create example input matrices
d__km = np.array([100, 200, 300, 400])
h_1__meter = np.array([10, 20, 30, 40])
h_2__meter = np.array([100, 200, 300, 400])
f__mhz = np.array([1000, 2000, 3000, 4000])
T_pol = np.array([0, 1, 0, 1])  # 0 for horizontal, 1 for vertical
p = np.array([50, 60, 70, 80])

# Call the P528 function with these matrices
return_value, result = P528(d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p)

# Now you can access the results
print("Propagation modes:", result.propagation_mode)
print("Path distances (km):", result.d__km)
print("Total losses (dB):", result.A__db)
print("Free-space losses (dB):", result.A_fs__db)
print("Atmospheric absorption losses (dB):", result.A_a__db)
print("Elevation angles (rad):", result.theta_h1__rad)
print("Result codes:", result.result)