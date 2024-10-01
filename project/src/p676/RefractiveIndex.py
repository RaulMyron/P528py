from p676 import *

def RefractiveIndex(p__hPa: float, T__kelvin: float, e__hPa: float) -> float:
    """
    Calculate the refractive index based on pressure, temperature, and water vapor pressure.

    Args:
    p__hPa (float): Pressure in hectopascals (hPa)
    T__kelvin (float): Temperature in Kelvin
    e__hPa (float): Water vapor pressure in hectopascals (hPa)

    Returns:
    float: Refractive index
    """
    # dry term of refractivity
    N_dry = 77.6 * p__hPa / T__kelvin

    # wet term of refractivity
    N_wet = 72 * e__hPa / T__kelvin + 3.75e5 * e__hPa / pow(T__kelvin, 2)

    N = N_dry + N_wet

    n = 1 + N * pow(10, -6)

    return n