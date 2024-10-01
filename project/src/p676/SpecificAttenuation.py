from p676 import *

def SpecificAttenuation(f__ghz: float, T__kelvin: float, e__hPa: float, p__hPa: float) -> float:
    """
    Calculate the specific attenuation due to atmospheric gases.

    Args:
    f__ghz (float): Frequency in GHz
    T__kelvin (float): Temperature in Kelvin
    e__hPa (float): Water vapor pressure in hectopascals (hPa)
    p__hPa (float): Total atmospheric pressure in hectopascals (hPa)

    Returns:
    float: Specific attenuation in dB/km
    """
    gamma_o = OxygenSpecificAttenuation(f__ghz, T__kelvin, e__hPa, p__hPa)

    gamma_w = WaterVapourSpecificAttenuation(f__ghz, T__kelvin, e__hPa, p__hPa)

    gamma = gamma_o + gamma_w   # [Eqn 1]

    return gamma

def OxygenSpecificAttenuation(f__ghz: float, T__kelvin: float, e__hPa: float, p__hPa: float) -> float:
    """
    Calculate the specific attenuation due to oxygen.

    Args:
    f__ghz (float): Frequency in GHz
    T__kelvin (float): Temperature in Kelvin
    e__hPa (float): Water vapor partial pressure in hectopascals (hPa)
    p__hPa (float): Dry air pressure in hectopascals (hPa)

    Returns:
    float: Specific attenuation due to oxygen in dB/km
    """
    # partial Eqn 1
    N_o = OxygenRefractivity(f__ghz, T__kelvin, e__hPa, p__hPa)
    
    gamma_o = 0.1820 * f__ghz * N_o

    return gamma_o

def WaterVapourSpecificAttenuation(f__ghz: float, T__kelvin: float, e__hPa: float, p__hPa: float) -> float:
    """
    Calculate the specific attenuation due to water vapour.

    Args:
    f__ghz (float): Frequency in GHz
    T__kelvin (float): Temperature in Kelvin
    e__hPa (float): Water vapor partial pressure in hectopascals (hPa)
    p__hPa (float): Dry air pressure in hectopascals (hPa)

    Returns:
    float: Specific attenuation due to water vapour in dB/km
    """
    # partial Eqn 1
    N_w = WaterVapourRefractivity(f__ghz, T__kelvin, e__hPa, p__hPa)
    gamma_w = 0.1820 * f__ghz * N_w

    return gamma_w