from line_profiler import LineProfiler
import sys
sys.path.append('/home/raulm/anatel/P528py/scalar')
from p528 import (P528, P528_Ex, ValidateInputs, TerminalGeometry, LineOfSight, 
                 RayOptics, GetPathLoss, ReflectionCoefficients, FindPsiAtDistance,
                 FindPsiAtDeltaR, FindDistanceAtDeltaR, SmoothEarthDiffraction,
                 TranshorizonSearch, troposcatter, LongTermVariability, 
                 NakagamiRice, CombineDistributions)

def run_tests():
    result = P528(0, 2, 3, 1000, 0, 50)
    return result

# Create profiler
profiler = LineProfiler()

# Add all relevant functions to profile
profiler.add_function(P528)
profiler.add_function(P528_Ex)
profiler.add_function(ValidateInputs)
profiler.add_function(TerminalGeometry)
profiler.add_function(LineOfSight)
profiler.add_function(RayOptics)
profiler.add_function(GetPathLoss)
profiler.add_function(ReflectionCoefficients)
profiler.add_function(FindPsiAtDistance)
profiler.add_function(FindPsiAtDeltaR)
profiler.add_function(FindDistanceAtDeltaR)
profiler.add_function(SmoothEarthDiffraction)
profiler.add_function(TranshorizonSearch)
profiler.add_function(troposcatter)
profiler.add_function(LongTermVariability)
profiler.add_function(NakagamiRice)
profiler.add_function(CombineDistributions)

# Wrap the test function
profiler_wrapper = profiler(run_tests)

# Run the profiler
profiler_wrapper()

# Save the results to a file
with open('line_profile_results.txt', 'w') as f:
    profiler.print_stats(stream=f)

print("Line profiling results have been saved to 'line_profile_results.txt'")