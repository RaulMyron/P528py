def ConvertToGeopotentialHeight(h__km: float) -> float:
    """
    Converts from geometric height, in km, to geopotential height, in km'.
    See Equation (1a).

    Args:
    h__km (float): Geometric height, in km

    Returns:
    float: Geopotential height, in km'
    """
    return (6356.766 * h__km) / (6356.766 + h__km)

def ConvertToGeometricHeight(h_prime__km: float) -> float:
    """
    Converts from geopotential height, in km', to geometric height, in km.
    See Equation (1b).

    Args:
    h_prime__km (float): Geopotential height, in km'

    Returns:
    float: Geometric height, in km
    """
    return (6356.766 * h_prime__km) / (6356.766 - h_prime__km)

def WaterVapourDensityToPressure(rho: float, T__kelvin: float) -> float:
    """
    Converts water vapour density, in g/m^3, to water vapour pressure, in hPa.
    See Equation (8).

    Args:
    rho (float): Water vapour density, rho(h), in g/m^3
    T__kelvin (float): Temperature, T(h), in Kelvin

    Returns:
    float: Water vapour pressure, e(h), in hPa
    """
    return (rho * T__kelvin) / 216.7

