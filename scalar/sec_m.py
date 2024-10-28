from p528 import P528

#print('------------------------------------------- 1')
#result = P528(0, 2, 3, 1000, 0, 50)
#print('------------------------------------------- 2')
#result = P528(200, 20, 200, 2000, 1, 60)
#print('------------------------------------------- 3')
#result = P528(300, 30, 300, 3000, 0, 70)
print('------------------------------------------- 4')
result = P528(400, 40, 20000, 4000, 1, 80)
#print(result)

print("Modo de propagação:", result.propagation_mode)
print("Perda total de transmissão:", result.A__db, "dB")
print("Perda no espaço livre:", result.A_fs__db, "dB")
print("Perda por absorção atmosférica:", result.A_a__db, "dB")
print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")