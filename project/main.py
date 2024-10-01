from p528 import P528

# Define the input values

d__km = float(input("Insira a distância (km): "))
h_1__meter = float(input("Insira a altura do terminal 1 (metros):"))
h_2__meter = float(input("Insira a altura do terminal 2 (metros):"))
f__mhz = float("Insira a frequência (MHz): ")
polarization = float(input("Insira a polarização (0 para horizontal, 1 para vertical): "))
time_percentage = float(input("Insira a porcentagem de tempo: "))

# Call the P528 function to calculate the basic transmission loss
result = P528(d__km, h_1__meter, h_2__meter, f__mhz, polarization, time_percentage)

# Display the results
print("Modo de propagação:", result.propagation_mode)
print("Perda total de transmissão:", result.A__db, "dB")
print("Perda no espaço livre:", result.A_fs__db, "dB")
print("Perda por absorção atmosférica:", result.A_a__db, "dB")
print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")
