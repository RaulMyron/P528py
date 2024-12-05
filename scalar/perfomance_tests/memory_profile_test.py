from memory_profiler import profile
import sys
sys.path.append('/home/raulm/anatel/P528py/scalar')
from p528 import (P528, P528_Ex, ValidateInputs, TerminalGeometry, LineOfSight, 
                 RayOptics, GetPathLoss, ReflectionCoefficients, FindPsiAtDistance,
                 FindPsiAtDeltaR, FindDistanceAtDeltaR, SmoothEarthDiffraction,
                 TranshorizonSearch, troposcatter, LongTermVariability, 
                 NakagamiRice, CombineDistributions)

# Profile main function with known working parameters
@profile
def test_P528_cases():
    print("\nTesting multiple P528 cases...")
    # These are known working cases
    cases = [
        (100.0, 10.0, 1000.0, 1000.0, 0, 50),  
        (200.0, 20.0, 200.0, 2000.0, 1, 60),   
        (300.0, 30.0, 300.0, 3000.0, 0, 70),   
        (400.0, 40.0, 20000.0, 4000.0, 1, 80)  
    ]
    
    results = []
    for case in cases:
        d_km, h1_m, h2_m, f_mhz, t_pol, p = case
        result = P528(d_km, h1_m, h2_m, f_mhz, t_pol, p)
        results.append(result)
        print(f"\nCase: d={d_km}km, h1={h1_m}m, h2={h2_m}m, f={f_mhz}MHz")
        print(f"Propagation Mode: {result.propagation_mode}")
        print(f"Total Loss: {result.A__db:.2f} dB")
    
    return results

@profile
def test_P528_Ex_case():
    print("\nTesting P528_Ex with components...")
    from p528 import Terminal, TroposcatterParams, Path, LineOfSightParams, Result
    
    # Use parameters from a known working case
    d_km = 100.0
    h1_m = 10.0
    h2_m = 1000.0
    f_mhz = 1000.0
    t_pol = 0
    p = 50
    
    terminal_1 = Terminal()
    terminal_2 = Terminal()
    tropo = TroposcatterParams()
    path = Path()
    los_params = LineOfSightParams()
    result = Result()
    
    return P528_Ex(d_km, h1_m, h2_m, f_mhz, t_pol, p, result, 
                   terminal_1, terminal_2, tropo, path, los_params)

def run_all_tests():
    print("Starting memory profiling tests...")
    print("="*50)
    
    # Run main test cases
    results = test_P528_cases()
    
    # Run extended test
    result_ex = test_P528_Ex_case()
    
    print("\nAll tests completed.")
    print("="*50)

if __name__ == '__main__':
    # Redirect output to file
    import sys
    original_stdout = sys.stdout
    
    try:
        with open('memory_profile_results.txt', 'w') as f:
            sys.stdout = f
            run_all_tests()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
    finally:
        # Always restore stdout
        sys.stdout = original_stdout
        print("Memory profiling results have been saved to 'memory_profile_results.txt'")