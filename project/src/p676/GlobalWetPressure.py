from p676 import *
from p835 import *

def GlobalWetPressure(h__km: float) -> float:
    """
    Calculate the global wet pressure at a given height.

    Args:
    h__km (float): Height in kilometers

    Returns:
    float: Wet pressure in hPa
    """
    T__kelvin = GlobalTemperature(h__km)
    P__hPa = GlobalPressure(h__km)
    rho__g_m3 = max(GlobalWaterVapourDensity(h__km, RHO_0__M_KG), 2 * pow(10, -6) * 216.7 * P__hPa / T__kelvin)
    e__hPa = WaterVapourDensityToPressure(rho__g_m3, T__kelvin)

    return e__hPa
