from p676 import *
from p835 import *
from RayTrace import *

def SlantPathAttenuation(f__ghz: float, h_1__km: float, h_2__km: float, beta_1__rad: float) -> SlantPathAttenuationResult:
    
    #config = RayTraceConfig(
    #    temperature=GlobalTemperature,
    #    dry_pressure=GlobalPressure,
    #    wet_pressure=GlobalWaterVapourPressure
    #)

    result = SlantPathAttenuationResult()

    if beta_1__rad > pi / 2:
        # negative elevation angle
        # find h_G and then trace in each direction
        # see Section 2.2.2

        # compute refractive index at h_1
        #p__hPa = config.dry_pressure(h_1__km)
        #T__kelvin = config.temperature(h_1__km)
        #e__hPa = config.wet_pressure(h_1__km, rho_0)  # Note: rho_0 needs to be defined or passed as a parameter

        p__hPa = GlobalPressure(h_1__km);
        T__kelvin = GlobalTemperature(h_1__km);
        e__hPa = GlobalWetPressure(h_1__km);
        
        n_1 = RefractiveIndex(p__hPa, T__kelvin, e__hPa)

        # set initial h_G at mid-point between h_1 and surface of the earth
        # then binary search to converge
        h_G__km = h_1__km
        delta = h_1__km / 2
        diff = 100

        while abs(diff) > 0.001:
            if diff > 0:
                h_G__km -= delta
            else:
                h_G__km += delta
            delta /= 2

            #p__hPa = config.dry_pressure(h_G__km)
            #T__kelvin = config.temperature(h_G__km)
            #e__hPa = config.wet_pressure(h_G__km, rho_0)  # Note: rho_0 needs to be defined or passed as a parameter

            p__hPa = GlobalPressure(h_G__km)
            T__kelvin = GlobalTemperature(h_G__km)
            e__hPa = GlobalWetPressure(h_G__km);
            
            n_G = RefractiveIndex(p__hPa, T__kelvin, e__hPa)

            grazing_term = n_G * (a_0__km + h_G__km)
            start_term = n_1 * (a_0__km + h_1__km) * math.sin(beta_1__rad)

            diff = grazing_term - start_term
        
        # converged on h_G.  Now call RayTrace in both directions with grazing angle
        beta_graze__rad = pi / 2
        result_1 = RayTrace(f__ghz, h_G__km, h_1__km, beta_graze__rad)
        result_2 = RayTrace(f__ghz, h_G__km, h_2__km, beta_graze__rad)

        result.angle__rad = result_2.angle__rad
        
        result.A_gas__db = result_1.A_gas__db + result_2.A_gas__db
        result.a__km = result_1.a__km + result_2.a__km
        result.bending__rad = result_1.bending__rad + result_2.bending__rad
        result.delta_L__km = result_1.delta_L__km + result_2.delta_L__km
    else:
        result = RayTrace(f__ghz, h_1__km, h_2__km, beta_1__rad)
        #result = RayTrace(f__ghz, h_1__km, h_2__km, beta_1__rad, config)

    return result