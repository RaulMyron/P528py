# flake8: noqa
from p528 import *

terminal_1 = Terminal()
terminal_2 = Terminal()
tropo = TroposcatterParams()
path = Path()
los_params = LineOfSightParams()
result = Result()

print('---------------------- main objects')

#result = P528_Ex(400, 40, 20000, 4000, 1, 80, result,
#                           terminal_1, terminal_2, tropo, path, los_params)


result = P528_Ex(0,1.5,1000,100,0,1, 
    result, terminal_1, terminal_2, tropo, path, los_params)


print("Modo de propagação:", result.propagation_mode)
print("Perda total de transmissão:", result.A__db, "dB")
print("Perda no espaço livre:", result.A_fs__db, "dB")
print("Perda por absorção atmosférica:", result.A_a__db, "dB")
print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")

result.clear(); terminal_1.clear(); terminal_2.clear(); tropo.clear(); path.clear(); los_params.clear()