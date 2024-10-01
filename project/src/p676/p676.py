from dataclasses import dataclass
from typing import Callable, List



# Constants
PI = 3.1415926535897932384
a_0__km = 6371.0

# Define function type aliases
Temperature = Callable[[float], float]
DryPressure = Callable[[float], float]
WetPressure = Callable[[float], float]

@dataclass
class SlantPathAttenuationResult:
    A_gas__db: float = 0.0        # Median gaseous absorption, in dB
    bending__rad: float = 0.0     # Bending angle, in rad
    a__km: float = 0.0            # Ray length, in km
    angle__rad: float = 0.0       # Incident angle, in rad
    delta_L__km: float = 0.0      # Excess atmospheric path length, in km

@dataclass
class RayTraceConfig:
    temperature: Temperature
    dry_pressure: DryPressure
    wet_pressure: WetPressure
    
class Result:
    propagation_mode: int = 0
    d__km: float = 0.0
    A__db: float = 0.0
    A_fs__db: float = 0.0
    A_a__db: float = 0.0
    theta_h1__rad: float = 0.0
    result: str = ''