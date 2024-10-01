import math
from p676 import *

def NonresonantDebyeAttenuation(f__ghz: float, e__hPa: float, p__hPa: float, theta: float) -> float:
    """
    Calculate the Non-resonant Debye component of frequency-dependent complex refractivity.

    Args:
    f__ghz (float): Frequency in GHz
    e__hPa (float): Water vapour partial pressure in hectopascals (hPa)
    p__hPa (float): Dry air pressure in hectopascals (hPa)
    theta (float): From Equation 3

    Returns:
    float: Non-resonant Debye component
    """
    # width parameter for the Debye spectrum, Equation 9
    d = 5.6e-4 * (p__hPa + e__hPa) * math.pow(theta, 0.8)

    # Equation 8
    frac_1 = 6.14e-5 / (d * (1 + math.pow(f__ghz / d, 2)))
    frac_2 = (1.4e-12 * p__hPa * math.pow(theta, 1.5)) / (1 + 1.9e-5 * math.pow(f__ghz, 1.5))
    N_D = f__ghz * p__hPa * math.pow(theta, 2) * (frac_1 + frac_2)

    return N_D