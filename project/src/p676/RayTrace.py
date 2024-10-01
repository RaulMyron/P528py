from p676 import *
from p835 import *

import math

def LayerThickness(m: float, i: int) -> float:
    # Equation 14
    delta_i__km = m * math.exp((i - 1) / 100.)
    return delta_i__km

def RayTrace(f__ghz: float, h_1__km: float, h_2__km: float, beta_1__rad: float) -> SlantPathAttenuationResult:
    # Equations 16(a)-(c)
    i_lower = math.floor(100 * math.log(1e4 * h_1__km * (math.exp(1. / 100.) - 1) + 1) + 1)
    i_upper = math.ceil(100 * math.log(1e4 * h_2__km * (math.exp(1. / 100.) - 1) + 1) + 1)
    m = ((math.exp(2. / 100.) - math.exp(1. / 100.)) / (math.exp(i_upper / 100.) - math.exp(i_lower / 100.))) * (h_2__km - h_1__km)

    result = SlantPathAttenuationResult()
    result.A_gas__db = 0
    result.bending__rad = 0
    result.a__km = 0
    result.delta_L__km = 0

    # initialize starting layer
    delta_i__km = LayerThickness(m, i_lower)
    h_i__km = h_1__km + m * ((math.exp((i_lower - 1) / 100.) - math.exp((i_lower - 1) / 100.)) / (math.exp(1 / 100.) - 1))
    n_i, gamma_i = GetLayerProperties(f__ghz, h_i__km + delta_i__km / 2)
    r_i__km = a_0__km + h_i__km
    
    # record bottom layer properties for alpha and beta calculations
    r_1__km = r_i__km
    n_1 = n_i

    # summation from Equation 13
    for i in range(i_lower, i_upper):
        delta_ii__km = LayerThickness(m, i + 1)
        h_ii__km = h_1__km + m * ((math.exp(i / 100.) - math.exp((i_lower - 1) / 100.)) / (math.exp(1 / 100.) - 1))

        n_ii, gamma_ii = GetLayerProperties(f__ghz, h_ii__km + delta_ii__km / 2)

        r_ii__km = a_0__km + h_ii__km

        delta_i__km = LayerThickness(m, i)

        # Equation 19b
        beta_i__rad = math.asin(min(1, (n_1 * r_1__km) / (n_i * r_i__km) * math.sin(beta_1__rad)))

        # entry angle into the layer interface, Equation 18a
        alpha_i__rad = math.asin(min(1, (n_1 * r_1__km) / (n_i * r_ii__km) * math.sin(beta_1__rad)))

        # path length through ith layer, Equation 17
        a_i__km = -r_i__km * math.cos(beta_i__rad) + math.sqrt(r_i__km**2 * math.cos(beta_i__rad)**2 + 2 * r_i__km * delta_i__km + delta_i__km**2)
        
        result.a__km += a_i__km
        result.A_gas__db += a_i__km * gamma_i
        result.delta_L__km += a_i__km * (n_i - 1)     # summation, Equation 23

        beta_ii__rad = math.asin(n_i / n_ii * math.sin(alpha_i__rad))

        # summation of the bending angle, Equation 22a
        # the summation only goes to i_max - 1
        if i != i_upper - 1:
            result.bending__rad += beta_ii__rad - alpha_i__rad

        # shift for next loop
        h_i__km = h_ii__km
        n_i = n_ii
        gamma_i = gamma_ii
        r_i__km = r_ii__km

    result.angle__rad = alpha_i__rad
    return result

def GetLayerProperties(f__ghz: float, h_i__km: float) -> tuple[float, float]:
    # use function pointers to get atmospheric parameters
    
    #T__kelvin = config.temperature(h_i__km)
    #p__hPa = config.dry_pressure(h_i__km)
    #e__hPa = config.wet_pressure(h_i__km)
    '''na original é assim, mas o certo em python seria:'''    
    T__kelvin = GlobalTemperature(h_i__km)
    p__hPa = GlobalPressure(h_i__km)
    e__hPa = GlobalWetPressure(h_i__km)
    
    # compute the refractive index for the current layer
    n = RefractiveIndex(p__hPa, T__kelvin, e__hPa)

    # specific attenuation of layer
    gamma = SpecificAttenuation(f__ghz, T__kelvin, e__hPa, p__hPa)

    return n, gamma