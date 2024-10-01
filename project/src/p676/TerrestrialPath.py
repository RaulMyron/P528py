# not used 

from SpecificAttenuation import *

def terrestrial_path(f_ghz: float, T_kelvin: float, e_hPa: float, p_hPa: float, r_0_km: float) -> float:
    """
    Calculate the terrestrial path attenuation.

    Args:
    f_ghz (float): Frequency in GHz
    T_kelvin (float): Temperature in Kelvin
    e_hPa (float): Water vapor partial pressure in hPa
    p_hPa (float): Atmospheric pressure in hPa
    r_0_km (float): Path length in km

    Returns:
    float: Attenuation in dB
    """
    gamma = SpecificAttenuation(f_ghz, T_kelvin, e_hPa, p_hPa)

    # Equation 10
    A_db = gamma * r_0_km

    return A_db