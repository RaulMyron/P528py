from typing import Callable, List

# Constants
PI = 3.1415926535897932384
a_0__km = 6371.0

# Function pointer types in Python
Temperature = Callable[[float], float]
DryPressure = Callable[[float], float]
WetPressure = Callable[[float], float]

class SlantPathAttenuationResult:
    def __init__(self):
        self.A_gas__db = 0.0  # Median gaseous absorption, in dB
        self.bending__rad = 0.0  # Bending angle, in rad
        self.a__km = 0.0  # Ray length, in km
        self.angle__rad = 0.0  # Incident angle, in rad
        self.delta_L__km = 0.0  # Excess atmospheric path length, in km

class RayTraceConfig:
    def __init__(self, temperature: Temperature, dry_pressure: DryPressure, wet_pressure: WetPressure):
        self.temperature = temperature
        self.dry_pressure = dry_pressure
        self.wet_pressure = wet_pressure

class OxygenData:
    f_0: List[float] = []
    a_1: List[float] = []
    a_2: List[float] = []
    a_3: List[float] = []
    a_4: List[float] = []
    a_5: List[float] = []
    a_6: List[float] = []

class WaterVapourData:
    f_0: List[float] = [] 
    b_1: List[float] = []
    b_2: List[float] = []
    b_3: List[float] = []
    b_4: List[float] = []
    b_5: List[float] = []
    b_6: List[float] = []