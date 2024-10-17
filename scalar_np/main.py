from p528 import P528

#d__km = 100.0
#h_1__meter = 10.0
#h_2__meter = 50.0
#f__mhz = 1000.0
#T_pol = 0  # horizontal polarization
#p = 50.0

#result = Result()  # Assuming you have a Result class defined
#result = P528(100.0, 10.0, 1000.0, 1000.0, 0, 50) #resultados bateram pra isso aqui

#print('-------------=------------------------------')

#print("Modo de propagação:", result.propagation_mode)
#print("Perda total de transmissão:", result.A__db, "dB")
#print("Perda no espaço livre:", result.A_fs__db, "dB")
#print("Perda por absorção atmosférica:", result.A_a__db, "dB")
#print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")

#print('-------------=------------------------------')

#result = P528(100.0, 10.0, 1000.0, 200000, 0, 50) #funciona
#print(result)

print('------------------------------------------- 1')
result = P528(0, 2, 3, 1000, 0, 50)
print("Modo de propagação:", result.propagation_mode)
print("Perda total de transmissão:", result.A__db, "dB")
print("Perda no espaço livre:", result.A_fs__db, "dB")
print("Perda por absorção atmosférica:", result.A_a__db, "dB")
print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")

print('------------------------------------------- 2')

result = P528(200, 20, 200, 2000, 1, 60)
print("Modo de propagação:", result.propagation_mode)
print("Perda total de transmissão:", result.A__db, "dB")
print("Perda no espaço livre:", result.A_fs__db, "dB")
print("Perda por absorção atmosférica:", result.A_a__db, "dB")
print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")

print('------------------------------------------- 3')

result = P528(300, 30, 300, 3000, 0, 70)
print("Modo de propagação:", result.propagation_mode)
print("Perda total de transmissão:", result.A__db, "dB")
print("Perda no espaço livre:", result.A_fs__db, "dB")
print("Perda por absorção atmosférica:", result.A_a__db, "dB")
print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")

print('------------------------------------------- 4')

result = P528(400, 40, 20000, 4000, 1, 80)
print("Modo de propagação:", result.propagation_mode)
print("Perda total de transmissão:", result.A__db, "dB")
print("Perda no espaço livre:", result.A_fs__db, "dB")
print("Perda por absorção atmosférica:", result.A_a__db, "dB")
print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")