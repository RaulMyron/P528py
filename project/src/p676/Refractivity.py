from p676 import *
from OxygenData import *
from LineShapeFactor import *
from NonresonantDebyeAttenuation import *
from WaterVapourData import *

import math

def OxygenRefractivity(f__ghz: float, T__kelvin: float, e__hPa: float, p__hPa: float) -> float:
    """
    Calculate the imaginary part of the frequency-dependent complex refractivity due to oxygen.

    Args:
    f__ghz (float): Frequency in GHz
    T__kelvin (float): Temperature in Kelvin
    e__hPa (float): Water vapour partial pressure in hectopascals (hPa)
    p__hPa (float): Dry air pressure in hectopascals (hPa)

    Returns:
    float: Refractivity in N-Units
    """
    theta = 300 / T__kelvin

    N = 0

    for i in range(len(OxygenData['f_0'])):
        # Equation 3, for oxygen
        S_i = OxygenData['a_1'][i] * 1e-7 * p__hPa * math.pow(theta, 3) * math.exp(OxygenData['a_2'][i] * (1 - theta))

        # compute the width of the line, Equation 6a, for oxygen
        delta_f__ghz = OxygenData['a_3'][i] * 1e-4 * (p__hPa * math.pow(theta, (0.8 - OxygenData['a_4'][i])) + 1.1 * e__hPa * theta)

        # modify the line width to account for Zeeman splitting of the oxygen lines
        # Equation 6b, for oxygen
        delta_f__ghz = math.sqrt(math.pow(delta_f__ghz, 2) + 2.25e-6)

        # correction factor due to interference effects in oxygen lines
        # Equation 7, for oxygen
        delta = (OxygenData['a_5'][i] + OxygenData['a_6'][i] * theta) * 1e-4 * (p__hPa + e__hPa) * math.pow(theta, 0.8)

        F_i = LineShapeFactor(f__ghz, OxygenData['f_0'][i], delta_f__ghz, delta)

        # summation of terms...from Equation 2a, for oxygen
        N += S_i * F_i

    N_D = NonresonantDebyeAttenuation(f__ghz, e__hPa, p__hPa, theta)

    N_o = N + N_D

    return N_o

def WaterVapourRefractivity(f__ghz: float, T__kelvin: float, e__hPa: float, P__hPa: float) -> float:
    """
    Calculate the imaginary part of the frequency-dependent complex refractivity due to water vapour.

    Args:
    f__ghz (float): Frequency in GHz
    T__kelvin (float): Temperature in Kelvin
    e__hPa (float): Water vapour partial pressure in hectopascals (hPa)
    P__hPa (float): Dry air pressure in hectopascals (hPa)

    Returns:
    float: Refractivity in N-Units
    """
    theta = 300 / T__kelvin

    N_w = 0
        
    for i in range(len(WaterVapourData['f_0'])):
        # Equation 3, for water vapour
        S_i = 0.1 * WaterVapourData['b_1'][i] * e__hPa * theta**3.5 * math.exp(WaterVapourData['b_2'][i] * (1 - theta))
        
        # compute the width of the line, Equation 6a, for water vapour
        delta_f__ghz = 1e-4 * WaterVapourData['b_3'][i] * (P__hPa * theta**WaterVapourData['b_4'][i] + WaterVapourData['b_5'][i] * e__hPa * theta**WaterVapourData['b_6'][i])
        
        # modify the line width to account for Doppler broadening of water vapour lines
        # Equation 6b, for water vapour
        term1 = 0.217 * delta_f__ghz**2 + (2.1316e-12 * WaterVapourData['f_0'][i]**2 / theta)
        delta_f__ghz = 0.535 * delta_f__ghz + math.sqrt(term1)
        
        # Equation 7, for water vapour
        delta = 0
        
        F_i = LineShapeFactor(f__ghz, WaterVapourData['f_0'][i], delta_f__ghz, delta)
        
        # summation of terms...from Equation 2b, for water vapour
        N_w += S_i * F_i

    return N_w