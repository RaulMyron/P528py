from p528 import *
import numpy as np
import random
from tqdm import tqdm

def generate_random_params():
    # Distance in km (greater than 0)
    d__km = random.uniform(0.1, 1000)  # Setting a reasonable upper limit
    
    # Heights in meters (1.5 to 20000)
    h_1__meter = random.uniform(1.5, 20000)
    h_2__meter = random.uniform(1.5, 20000)
    
    # Frequency in MHz (100 to 30000)
    f__mhz = random.uniform(100, 30000)
    
    # Polarization (0 or 1)
    T_pol = random.randint(0, 1)
    
    # Time percentage (1 to 99)
    p = random.uniform(1, 99)
    
    return d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p

def test_p528(num_tests=1000000):
    # Initialize counters
    successful_tests = 0
    failed_tests = 0
    
    # Initialize objects
    terminal_1 = Terminal()
    terminal_2 = Terminal()
    tropo = TroposcatterParams()
    path = Path()
    los_params = LineOfSightParams()
    result = Result()
    los_result = LineOfSightParams()
    
    # Run tests with progress bar
    for _ in tqdm(range(num_tests), desc="Running tests"):
        try:
            # Generate random parameters
            d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p = generate_random_params()
            
            # Run P528_Ex
            P528_Ex(d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p, result,
                   terminal_1, terminal_2, tropo, path, los_params, los_result)
            
            successful_tests += 1
            
        except Exception as e:
            failed_tests += 1
            if failed_tests <= 5:  # Only print first 5 errors to avoid flooding console
                print(f"Error in test: {e}")
                print(f"Parameters: d={d__km}km, h1={h_1__meter}m, h2={h_2__meter}m, f={f__mhz}MHz, T_pol={T_pol}, p={p}%")
        
        # Clear objects after each test
        result.clear()
        terminal_1.clear()
        terminal_2.clear()
        tropo.clear()
        path.clear()
        los_params.clear()
        los_result.clear()
    
    return successful_tests, failed_tests

if __name__ == "__main__":
    # Number of tests to run
    NUM_TESTS = 100  # You can modify this value
    
    print(f"Starting {NUM_TESTS} tests...")
    successful, failed = test_p528(NUM_TESTS)
    
    print("\nTest Results:")
    print(f"Total tests: {NUM_TESTS}")
    print(f"Successful tests: {successful}")
    print(f"Failed tests: {failed}")
    print(f"Success rate: {(successful/NUM_TESTS)*100:.2f}%")