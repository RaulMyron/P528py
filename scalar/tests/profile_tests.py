import cProfile
import pstats
from pstats import SortKey
import sys
sys.path.append('/home/raulm/anatel/P528py/scalar')
from p528 import P528

def run_tests():

    #fazer uns 1000 testes

    result = P528(400, 40, 20000, 4000, 1, 80)
    
    #print("Modo de propagação:", result.propagation_mode)
    #print("Perda total de transmissão:", result.A__db, "dB")
    #print("Perda no espaço livre:", result.A_fs__db, "dB")
    #print("Perda por absorção atmosférica:", result.A_a__db, "dB")
    #print("Ângulo de elevação no terminal 1:", result.theta_h1__rad, "rad")

# Create a Profile object
profiler = cProfile.Profile()

# Start profiling
profiler.enable()

# Run the tests
run_tests()

# Stop profiling
profiler.disable()

# Create stats object and sort by cumulative time
stats = pstats.Stats(profiler)

# Save stats to a file
stats.dump_stats("profile_output.stats")

# Also save a readable text version
with open("profile_results.txt", "w") as f:
    # Print top 30 functions by cumulative time
    stats.sort_stats(SortKey.CUMULATIVE)
    f.write("\nTop 30 functions by cumulative time:\n")
    stats.stream = f  # Redirect output to file
    stats.print_stats(30)
    
    # Print top 30 functions by total time
    stats.sort_stats(SortKey.TIME)
    f.write("\nTop 30 functions by total time:\n")
    stats.print_stats(30)

print("Profiling results have been saved to 'profile_results.txt'")
print("Raw stats have been saved to 'profile_output.stats'")