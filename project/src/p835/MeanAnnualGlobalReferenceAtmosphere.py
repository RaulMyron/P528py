from p835 import *
from Conversions import *
import math

def GlobalTemperature(h__km: float) -> float:
    if h__km < 0:
        raise ValueError(ERROR_HEIGHT_TOO_SMALL)
    if h__km > 100:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)
    if h__km < 86:
        h_prime__km = ConvertToGeopotentialHeight(h__km)
        return GlobalTemperature_Regime1(h_prime__km)
    else:
        return GlobalTemperature_Regime2(h__km)

def GlobalTemperature_Regime1(h_prime__km: float) -> float:
    if h_prime__km < 0:
        raise ValueError(ERROR_HEIGHT_TOO_SMALL)
    elif h_prime__km <= 11:
        return 288.15 - 6.5 * h_prime__km
    elif h_prime__km <= 20:
        return 216.65
    elif h_prime__km <= 32:
        return 216.65 + (h_prime__km - 20)
    elif h_prime__km <= 47:
        return 228.65 + 2.8 * (h_prime__km - 32)
    elif h_prime__km <= 51:
        return 270.65
    elif h_prime__km <= 71:
        return 270.65 - 2.8 * (h_prime__km - 51)
    elif h_prime__km <= 84.852:
        return 214.65 - 2.0 * (h_prime__km - 71)
    else:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)

def GlobalTemperature_Regime2(h__km: float) -> float:
    if h__km < 86:
        return ERROR_HEIGHT_TOO_SMALL
    elif h__km <= 91:
        return 186.8673
    elif h__km <= 100:
        return 263.1905 - 76.3232 * math.sqrt(1 - pow((h__km - 91) / 19.9429, 2))
    else:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)

def GlobalTemperature_Regime2(h__km: float) -> float:
    if h__km < 86:
        return ERROR_HEIGHT_TOO_SMALL
    elif h__km <= 91:
        return 186.8673
    elif h__km <= 100:
        return 263.1905 - 76.3232 * math.sqrt(1 - pow((h__km - 91) / 19.9429, 2))
    else:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)
    
def GlobalPressure(h__km: float) -> float:
    if h__km < 0:
        raise ValueError(ERROR_HEIGHT_TOO_SMALL)
    if h__km > 100:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)

    if h__km < 86:
        h_prime__km = ConvertToGeopotentialHeight(h__km)
        return GlobalPressure_Regime1(h_prime__km)
    else:
        return GlobalPressure_Regime2(h__km)
    
def GlobalPressure_Regime1(h_prime__km: float) -> float:
    if h_prime__km < 0:
        raise ValueError(ERROR_HEIGHT_TOO_SMALL)
    elif h_prime__km <= 11:
        return 1013.25 * pow(288.15 / (288.15 - 6.5 * h_prime__km), -34.1632 / 6.5)
    elif h_prime__km <= 20:
        return 226.3226 * math.exp(-34.1632 * (h_prime__km - 11) / 216.65)
    elif h_prime__km <= 32:
        return 54.74980 * pow(216.65 / (216.65 + (h_prime__km - 20)), 34.1632)
    elif h_prime__km <= 47:
        return 8.680422 * pow(228.65 / (228.65 + 2.8 * (h_prime__km - 32)), 34.1632 / 2.8)
    elif h_prime__km <= 51:
        return 1.109106 * math.exp(-34.1632 * (h_prime__km - 47) / 270.65)
    elif h_prime__km <= 71:
        return 0.6694167 * pow(270.65 / (270.65 - 2.8 * (h_prime__km - 51)), -34.1632 / 2.8)
    elif h_prime__km <= 84.852:
        return 0.03956649 * pow(214.65 / (214.65 - 2.0 * (h_prime__km - 71)), -34.1632 / 2.0)
    else:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)
    
def GlobalPressure_Regime2(h__km: float) -> float:
    if h__km < 86:
        raise ValueError(ERROR_HEIGHT_TOO_SMALL)
    if h__km > 100:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)
    
    a_0 = 95.571899
    a_1 = -4.011801
    a_2 = 6.424731e-2
    a_3 = -4.789660e-4
    a_4 = 1.340543e-6

    return math.exp(a_0 + a_1 * h__km + a_2 * pow(h__km, 2) + a_3 * pow(h__km, 3) + a_4 * pow(h__km, 4))

def GlobalWaterVapourDensity(h__km: float, rho_0: float) -> float:
    if h__km < 0:
        raise ValueError(ERROR_HEIGHT_TOO_SMALL)
    if h__km > 100:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)

    h_0__km = 2  # scale height

    return rho_0 * math.exp(-h__km / h_0__km)

def GlobalWaterVapourPressure(h__km: float, rho_0: float) -> float:
    if h__km < 0:
        raise ValueError(ERROR_HEIGHT_TOO_SMALL)
    if h__km > 100:
        raise ValueError(ERROR_HEIGHT_TOO_LARGE)

    rho = GlobalWaterVapourDensity(h__km, rho_0)

    if h__km < 86:
        # convert to geopotential height
        h_prime__km = ConvertToGeopotentialHeight(h__km)
        T__kelvin = GlobalTemperature_Regime1(h_prime__km)
    else:
        T__kelvin = GlobalTemperature_Regime2(h__km)
    
    return WaterVapourDensityToPressure(rho, T__kelvin)