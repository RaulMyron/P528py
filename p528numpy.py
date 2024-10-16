import math
import numpy as np
from scipy import special
from copy import deepcopy

#P258.h

from dataclasses import dataclass, field
from typing import List

# Constants
PI = np.pi
a_0__km = 6371.0
a_e__km = 9257.0
N_s = 341
epsilon_r = 15.0
sigma = 0.005
LOS_EPSILON = 1e-5
THIRD = 1.0 / 3.0

# Consts
CONST_MODE__SEARCH = 0
CONST_MODE__DIFFRACTION = 1
CONST_MODE__SCATTERING = 2

CASE_1 = 1
CASE_2 = 2

PROP_MODE__NOT_SET = 0
PROP_MODE__LOS = 1
PROP_MODE__DIFFRACTION = 2
PROP_MODE__SCATTERING = 3

# List of valid polarizations
POLARIZATION__HORIZONTAL = 0
POLARIZATION__VERTICAL = 1

Y_pi_99_INDEX = 16

# Return Codes
SUCCESS = 0
ERROR_VALIDATION__D_KM = 1
ERROR_VALIDATION__H_1 = 2
ERROR_VALIDATION__H_2 = 3
ERROR_VALIDATION__TERM_GEO = 4
ERROR_VALIDATION__F_MHZ_LOW = 5
ERROR_VALIDATION__F_MHZ_HIGH = 6
ERROR_VALIDATION__PERCENT_LOW = 7
ERROR_VALIDATION__PERCENT_HIGH = 8
ERROR_VALIDATION__POLARIZATION = 9
ERROR_HEIGHT_AND_DISTANCE = 10
WARNING__DFRAC_TROPO_REGION = 20

# Classes

@dataclass
class Path:
    d_ML__km: np.ndarray[float] = np.array([0.0])
    d_0__km: np.ndarray[float] = np.array([0.0])
    d_d__km: np.ndarray[float] = np.array([0.0])

@dataclass
class Terminal:
    h_r__km: np.ndarray[float] = np.array([0.0])
    h_e__km: np.ndarray[float] = np.array([0.0])
    delta_h__km: np.ndarray[float] = np.array([0.0])
    d_r__km: np.ndarray[float] = np.array([0.0])
    a__km: np.ndarray[float] = np.array([0.0])
    phi__rad: np.ndarray[float] = np.array([0.0])
    theta__rad: np.ndarray[float] = np.array([0.0])
    A_a__db: np.ndarray[float] = np.array([0.0])

@dataclass
class LineOfSightParams:
    z__km: np.ndarray[float] = np.array([0.0, 0.0])
    d__km: np.ndarray[float] = np.array([0.0])
    r_0__km: np.ndarray[float] = np.array([0.0])
    r_12__km: np.ndarray[float] = np.array([0.0])
    D__km: np.ndarray[float] = np.array([0.0, 0.0])
    theta_h1__rad: np.ndarray[float] = np.array([0.0])
    theta_h2__rad: np.ndarray[float] = np.array([0.0])
    theta: np.ndarray[float] = np.array([0.0, 0.0])
    a_a__km: np.ndarray[float] = np.array([0.0])
    delta_r__km: np.ndarray[float] = np.array([0.0])
    A_LOS__db: np.ndarray[float] = np.array([0.0])

@dataclass
class TroposcatterParams:
    d_s__km: np.ndarray[float] = np.array([0.0])
    d_z__km: np.ndarray[float] = np.array([0.0])
    h_v__km: np.ndarray[float] = np.array([0.0])
    theta_s: np.ndarray[float] = np.array([0.0])
    theta_A: np.ndarray[float] = np.array([0.0])
    A_s__db: np.ndarray[float] = np.array([0.0])
    A_s_prev__db: np.ndarray[float] = np.array([0.0])
    M_s: np.ndarray[float] = np.array([0.0])

@dataclass
class Result:
    def __init__(self, size=1):
        self.propagation_mode = np.zeros(size, dtype=int)
        self.d__km = np.zeros(size, dtype=float)
        self.A__db = np.zeros(size, dtype=float)
        self.A_fs__db = np.zeros(size, dtype=float)
        self.A_a__db = np.zeros(size, dtype=float)
        self.theta_h1__rad = np.zeros(size, dtype=float)
        self.result = np.empty(size, dtype=object)

    def reset(self, size=None):
        if size is not None:
            self.__init__(size)
        else:
            self.A_fs__db.fill(0.0)
            self.A_a__db.fill(0.0)
            self.A__db.fill(0.0)
            self.d__km.fill(0.0)
            self.theta_h1__rad.fill(0.0)
            self.propagation_mode.fill(PROP_MODE__NOT_SET)
            
#P676.h

from dataclasses import dataclass
from typing import Callable, List

# Constants
a_0__km = 6371.0

@dataclass
class SlantPathAttenuationResult:
    A_gas__db: float = 0.0     # Median gaseous absorption, in dB
    bending__rad: float = 0.0  # Bending angle, in rad
    a__km: float = 0.0         # Ray length, in km
    angle__rad: float = 0.0    # Incident angle, in rad
    delta_L__km: float = 0.0   # Excess atmospheric path length, in km

#P835.h

# Constants
RHO_0__M_KG = 7.5

# Error Codes
ERROR_HEIGHT_TOO_SMALL = -1
ERROR_HEIGHT_TOO_LARGE = -2

#P528.cpp

# Constants
PROP_MODE__NOT_SET = 0
PROP_MODE__LOS = 1
PROP_MODE__DIFFRACTION = 2
PROP_MODE__SCATTERING = 3

THIRD = 1.0 / 3.0
SUCCESS = 0
ERROR_HEIGHT_AND_DISTANCE = -1
CASE_1 = 1
CASE_2 = 2
a_e__km = 9257.0

def P528(d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p):

    terminal_1 = Terminal()
    terminal_2 = Terminal()
    tropo = TroposcatterParams()
    path = Path()
    los_params = LineOfSightParams()
    result = Result()
    
    return_value = P528_Ex(d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p, result,
                           terminal_1, terminal_2, tropo, path, los_params)

    return return_value

def P528_Ex(d__km: np.ndarray[float], h_1__meter: np.ndarray[float], h_2__meter: np.ndarray[float], 
            f__mhz: np.ndarray[float], T_pol: np.ndarray[int], p: np.ndarray[float], 
            result: Result, terminal_1: Terminal, terminal_2: Terminal,
            tropo: TroposcatterParams, path: Path, los_params: LineOfSightParams) -> np.ndarray[int]:
    
    # reset Results struct
    size = len(d__km)
    result.reset(size)

    validation_result, error_messages = ValidateInputs(d__km, h_1__meter, h_2__meter, f__mhz, T_pol, p)

    result.result = validation_result
    
    if error_messages:
        print("Validation Error(s):")
        for error in error_messages:
            print(error)
        return Result

    # Compute terminal geometries
    
    # Step 1 for low terminal
    
    terminal_1.h_r__km = h_1__meter / 1000
    TerminalGeometry(f__mhz, terminal_1) #certo
    
    # Step 1 for high terminal
    terminal_2.h_r__km = h_2__meter / 1000
    TerminalGeometry(f__mhz, terminal_2) #certo
    
    # Step 2
    path.d_ML__km = terminal_1.d_r__km + terminal_2.d_r__km  # [Eqn 3-1]

    # Smooth earth diffraction line calculations

    # Step 3.1
    d_3__km = path.d_ML__km + 0.5 * np.power(np.power(a_e__km, 2) / f__mhz, THIRD)  # [Eqn 3-2]
    d_4__km = path.d_ML__km + 1.5 * np.power(np.power(a_e__km, 2) / f__mhz, THIRD)  # [Eqn 3-3]

    # Step 3.2
    A_3__db = SmoothEarthDiffraction(terminal_1.d_r__km, terminal_2.d_r__km, f__mhz, d_3__km, T_pol)
    A_4__db = SmoothEarthDiffraction(terminal_1.d_r__km, terminal_2.d_r__km, f__mhz, d_4__km, T_pol)

    # Step 3.3
    M_d = (A_4__db - A_3__db) / (d_4__km - d_3__km)  # [Eqn 3-4]
    A_d0 = A_4__db - M_d * d_4__km  # [Eqn 3-5]

    # Step 3.4
    A_dML__db = (M_d * path.d_ML__km) + A_d0  # [Eqn 3-6]
    path.d_d__km = -(A_d0 / M_d)  # [Eqn 3-7]

    # Determine LOS condition
    los_condition = path.d_ML__km - d__km > 0.001
    
    # Initialize K_LOS array
    K_LOS = np.zeros_like(d__km)
    
    # Handle LOS and non-LOS cases
    result.propagation_mode = np.where(los_condition, PROP_MODE__LOS, result.propagation_mode)
    
    # LOS case
    los_indices = np.where(los_condition)[0]
    if len(los_indices) > 0:
        los_path = index_object(path, los_indices)
        los_terminal_1 = index_object(terminal_1, los_indices)        
        los_terminal_2 = index_object(terminal_2, los_indices)        
        los_los_params = index_object(los_params, los_indices)
        
        los_result = LineOfSight(los_path, los_terminal_1, los_terminal_2, los_los_params, 
                                 f__mhz[los_indices], -A_dML__db[los_indices], 
                                 p[los_indices], d__km[los_indices], 
                                 T_pol[los_indices], result, K_LOS[los_indices])
        
        print('los, result', los_result)
        exit()
        
        # Update results for LOS cases
        for attr in dir(los_result):
            if not attr.startswith('__') and hasattr(result, attr):
                getattr(result, attr)[los_indices] = getattr(los_result, attr)
    
    # Non-LOS case
    non_los_indices = np.where(~los_condition)[0]
    if len(non_los_indices) > 0:
        non_los_path = index_object(path, non_los_indices)
        non_los_terminal_1 = index_object(terminal_1, non_los_indices)
        non_los_terminal_2 = index_object(terminal_2, non_los_indices)
        non_los_los_params = index_object(los_params, non_los_indices)
        
        K_LOS[non_los_indices] = LineOfSight(non_los_path, non_los_terminal_1, non_los_terminal_2, non_los_los_params,
                                             f__mhz[non_los_indices], -A_dML__db[non_los_indices], 
                                             p[non_los_indices], non_los_path.d_ML__km - 1, 
                                             T_pol[non_los_indices], result, K_LOS[non_los_indices])
        
        # Transhorizon search
        M_d, A_d0, d_crx__km, CASE = TranshorizonSearch(non_los_path, non_los_terminal_1, non_los_terminal_2, 
                                                        f__mhz[non_los_indices], A_dML__db[non_los_indices], 
                                                        M_d[non_los_indices], A_d0[non_los_indices])
        
        # Compute terrain attenuation, A_T__db
        A_d__db = M_d * d__km[non_los_indices] + A_d0  # [Eqn 3-14]
        
        # Troposcatter calculations
        non_los_tropo = index_object(tropo, non_los_indices)
        tropo_result = troposcatter(non_los_path, non_los_terminal_1, non_los_terminal_2, 
                                    d__km[non_los_indices], f__mhz[non_los_indices], non_los_tropo)
        
        # Determine propagation mode and attenuation
        diffraction_condition = d__km[non_los_indices] < d_crx__km
        A_T__db = np.where(diffraction_condition, A_d__db, np.minimum(tropo_result.A_s__db, A_d__db))
        result.propagation_mode[non_los_indices] = np.where(diffraction_condition, 
                                                            PROP_MODE__DIFFRACTION, 
                                                            np.where(tropo_result.A_s__db <= A_d__db, 
                                                                     PROP_MODE__SCATTERING, 
                                                                     PROP_MODE__DIFFRACTION))
        
        # Compute variability
        f_theta_h = np.ones_like(d__km[non_los_indices])
        Y_e__db, _ = LongTermVariability(non_los_terminal_1.d_r__km, non_los_terminal_2.d_r__km, 
                                         d__km[non_los_indices], f__mhz[non_los_indices], 
                                         p[non_los_indices], f_theta_h, -A_T__db)
        Y_e_50__db, _ = LongTermVariability(non_los_terminal_1.d_r__km, non_los_terminal_2.d_r__km, 
                                            d__km[non_los_indices], f__mhz[non_los_indices], 
                                            50, f_theta_h, -A_T__db)
        
        # Compute Nakagami-Rice distribution
        ANGLE = 0.02617993878  # 1.5 deg
        K_t__db = np.where(tropo_result.theta_s >= ANGLE, 20,
                           np.where(tropo_result.theta_s <= 0.0, K_LOS[non_los_indices],
                                    (tropo_result.theta_s * (20.0 - K_LOS[non_los_indices]) / ANGLE) + K_LOS[non_los_indices]))
        
        Y_pi_50__db = np.zeros_like(d__km[non_los_indices])
        Y_pi__db = NakagamiRice(K_t__db, p[non_los_indices])
        
        # Combine distributions
        Y_total__db = CombineDistributions(Y_e_50__db, Y_e__db, Y_pi_50__db, Y_pi__db, p[non_los_indices])
        
        # Atmospheric absorption for transhorizon path
        result_v = SlantPathAttenuation(f__mhz[non_los_indices] / 1000, 0, tropo_result.h_v__km, np.pi / 2)
        result.A_a__db[non_los_indices] = non_los_terminal_1.A_a__db + non_los_terminal_2.A_a__db + 2 * result_v.A_gas__db  # [Eqn 3-17]
        
        # Compute free-space loss
        r_fs__km = non_los_terminal_1.a__km + non_los_terminal_2.a__km + 2 * result_v.a__km  # [Eqn 3-18]
        result.A_fs__db[non_los_indices] = 20.0 * np.log10(f__mhz[non_los_indices]) + 20.0 * np.log10(r_fs__km) + 32.45  # [Eqn 3-19]
        
        # Final results
        result.d__km[non_los_indices] = d__km[non_los_indices]
        result.A__db[non_los_indices] = result.A_fs__db[non_los_indices] + result.A_a__db[non_los_indices] + A_T__db - Y_total__db  # [Eqn 3-20]
        result.theta_h1__rad[non_los_indices] = -non_los_terminal_1.theta__rad
    
    return result

def index_object(obj, indices):
    if isinstance(obj, LineOfSightParams):
        # Special case for LineOfSightParams
        return LineOfSightParams(
            z__km=np.zeros((len(indices), 2)),
            d__km=np.zeros(len(indices)),
            r_0__km=np.zeros(len(indices)),
            r_12__km=np.zeros(len(indices)),
            D__km=np.zeros((len(indices), 2)),
            theta_h1__rad=np.zeros(len(indices)),
            theta_h2__rad=np.zeros(len(indices)),
            theta=np.zeros((len(indices), 2)),
            a_a__km=np.zeros(len(indices)),
            delta_r__km=np.zeros(len(indices)),
            A_LOS__db=np.zeros(len(indices))
        )
    
    new_obj = deepcopy(obj)
    for attr, value in vars(new_obj).items():
        if isinstance(value, np.ndarray):
            if value.size == 1:
                # If it's a single-element array, duplicate it for all indices
                setattr(new_obj, attr, np.full(len(indices), value[0]))
            else:
                # If it's a multi-element array, index it normally
                setattr(new_obj, attr, value[indices])
    return new_obj

def ValidateInputs(d_km, h_1_meter, h_2_meter, f_mhz, t_pol, p):
    # Ensure all inputs are numpy arrays
    d_km, h_1_meter, h_2_meter, f_mhz, t_pol, p = map(np.atleast_1d, [d_km, h_1_meter, h_2_meter, f_mhz, t_pol, p])
    
    # Ensure all arrays have the same shape
    shape = np.broadcast_shapes(d_km.shape, h_1_meter.shape, h_2_meter.shape, f_mhz.shape, t_pol.shape, p.shape)
    d_km, h_1_meter, h_2_meter, f_mhz, t_pol, p = map(lambda x: np.broadcast_to(x, shape), [d_km, h_1_meter, h_2_meter, f_mhz, t_pol, p])

    # Create a result array
    result = np.full(shape, SUCCESS, dtype=int)

    # Create masks for all conditions
    masks = {
        'd_km': d_km < 0,
        'h_1_meter': (h_1_meter < 1.5) | (h_1_meter > 20000),
        'h_2_meter': (h_2_meter < 1.5) | (h_2_meter > 20000),
        'h_1_gt_h_2': h_1_meter > h_2_meter,
        'f_mhz': (f_mhz < 100) | (f_mhz > 30000),
        't_pol': (t_pol != POLARIZATION__HORIZONTAL) & (t_pol != POLARIZATION__VERTICAL),
        'p': (p < 1) | (p > 99),
        'equal_values': (h_1_meter == h_2_meter) & (h_1_meter == d_km) & (d_km != 0)
    }

    # Prepare error messages
    error_messages = []
    error_texts = {
        'd_km': "d_km must be non-negative",
        'h_1_meter': "h_1_meter must be between 1.5 and 20000",
        'h_2_meter': "h_2_meter must be between 1.5 and 20000",
        'h_1_gt_h_2': "h_1_meter must not be greater than h_2_meter",
        'f_mhz': "f_mhz must be between 100 and 30000",
        't_pol': "t_pol must be either 0 (horizontal) or 1 (vertical)",
        'p': "p must be between 1 and 99",
        'equal_values': "h_1_meter, h_2_meter, and d_km are equal (and non-zero)"
    }

    for key, mask in masks.items():
        if np.any(mask):
            invalid_indices = np.where(mask)[0]
            if key == 'h_1_gt_h_2':
                values = list(zip(h_1_meter[mask], h_2_meter[mask]))
            elif key == 'equal_values':
                values = list(zip(h_1_meter[mask], h_2_meter[mask], d_km[mask]))
            else:
                values = locals()[key][mask]
            error_messages.append(f"{error_texts[key]}. At (indices, values): {list(zip(invalid_indices+1, values))}")
            result[mask] = ERROR_HEIGHT_AND_DISTANCE if key == 'equal_values' else ERROR_VALIDATION__D_KM

    return result, error_messages

def TerminalGeometry(f__mhz: np.ndarray, terminal: Terminal) -> None:
    
    theta_tx__rad = np.zeros_like(f__mhz)
    
    result = SlantPathAttenuation(f__mhz / 1000, np.zeros_like(f__mhz), terminal.h_r__km, np.pi / 2 - theta_tx__rad)
    
    terminal.theta__rad = np.pi / 2 - result.angle__rad
    terminal.A_a__db = result.A_gas__db
    terminal.a__km = result.a__km
    
    # compute arc distance
    central_angle = ((np.pi / 2 - result.angle__rad) - theta_tx__rad + result.bending__rad)
    terminal.d_r__km = a_0__km * central_angle
    terminal.phi__rad = terminal.d_r__km / a_e__km
    terminal.h_e__km = (a_e__km / np.cos(terminal.phi__rad)) - a_e__km
    terminal.delta_h__km = terminal.h_r__km - terminal.h_e__km
    
def GlobalTemperature(h__km):
    result = np.zeros_like(h__km)
    for i in range(len(h__km)):
        if h__km[i] < 0:
            result[i] = ERROR_HEIGHT_TOO_SMALL
        elif h__km[i] > 100:
            result[i] = ERROR_HEIGHT_TOO_LARGE
        elif h__km[i] < 86:
            h_prime__km = ConvertToGeopotentialHeight(h__km[i])
            result[i] = GlobalTemperature_Regime1(h_prime__km)
        else:
            result[i] = GlobalTemperature_Regime2(h__km[i])
    return result

def GlobalTemperature_Regime1(h_prime__km):
    if h_prime__km <= 11:
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
        return ERROR_HEIGHT_TOO_LARGE

def GlobalTemperature_Regime2(h__km):
    if h__km < 86:
        return ERROR_HEIGHT_TOO_SMALL
    elif h__km <= 91:
        return 186.8673
    elif h__km <= 100:
        return 263.1905 - 76.3232 * np.sqrt(1 - ((h__km - 91) / 19.9429)**2)
    else:
        return ERROR_HEIGHT_TOO_LARGE

def GlobalPressure(h__km):
    result = np.zeros_like(h__km)
    for i in range(len(h__km)):
        if h__km[i] < 0:
            result[i] = ERROR_HEIGHT_TOO_SMALL
        elif h__km[i] > 100:
            result[i] = ERROR_HEIGHT_TOO_LARGE
        elif h__km[i] < 86:
            h_prime__km = ConvertToGeopotentialHeight(h__km[i])
            result[i] = GlobalPressure_Regime1(h_prime__km)
        else:
            result[i] = GlobalPressure_Regime2(h__km[i])
    return result

def GlobalPressure_Regime1(h_prime__km):
    if h_prime__km <= 11:
        return 1013.25 * (288.15 / (288.15 - 6.5 * h_prime__km))**(-34.1632 / 6.5)
    elif h_prime__km <= 20:
        return 226.3226 * np.exp(-34.1632 * (h_prime__km - 11) / 216.65)
    elif h_prime__km <= 32:
        return 54.74980 * (216.65 / (216.65 + (h_prime__km - 20)))**34.1632
    elif h_prime__km <= 47:
        return 8.680422 * (228.65 / (228.65 + 2.8 * (h_prime__km - 32)))**(34.1632 / 2.8)
    elif h_prime__km <= 51:
        return 1.109106 * np.exp(-34.1632 * (h_prime__km - 47) / 270.65)
    elif h_prime__km <= 71:
        return 0.6694167 * (270.65 / (270.65 - 2.8 * (h_prime__km - 51)))**(-34.1632 / 2.8)
    elif h_prime__km <= 84.852:
        return 0.03956649 * (214.65 / (214.65 - 2.0 * (h_prime__km - 71)))**(-34.1632 / 2.0)
    else:
        return ERROR_HEIGHT_TOO_LARGE

def GlobalPressure_Regime2(h__km):
    if h__km < 86:
        return ERROR_HEIGHT_TOO_SMALL
    elif h__km <= 100:
        a_0 = 95.571899
        a_1 = -4.011801
        a_2 = 6.424731e-2
        a_3 = -4.789660e-4
        a_4 = 1.340543e-6
        return np.exp(a_0 + a_1 * h__km + a_2 * h__km**2 + a_3 * h__km**3 + a_4 * h__km**4)
    else:
        return ERROR_HEIGHT_TOO_LARGE

def GlobalWaterVapourDensity(h__km, rho_0):
    result = np.zeros_like(h__km)
    h_0__km = 2  # scale height

    #np.where(<0, re

    for i in range(len(h__km)):
        if h__km[i] < 0:
            result[i] = ERROR_HEIGHT_TOO_SMALL
        elif h__km[i] > 100:
            result[i] = ERROR_HEIGHT_TOO_LARGE
        else:
            result[i] = rho_0 * np.exp(-h__km[i] / h_0__km)
    return result

def GlobalWaterVapourPressure(h__km, rho_0):
    result = np.zeros_like(h__km)
    for i in range(len(h__km)):
        if h__km[i] < 0:
            result[i] = ERROR_HEIGHT_TOO_SMALL
        elif h__km[i] > 100:
            result[i] = ERROR_HEIGHT_TOO_LARGE
        else:
            rho = GlobalWaterVapourDensity(np.array([h__km[i]]), rho_0)[0]
            if h__km[i] < 86:
                h_prime__km = ConvertToGeopotentialHeight(h__km[i])
                T__kelvin = GlobalTemperature_Regime1(h_prime__km)
            else:
                T__kelvin = GlobalTemperature_Regime2(h__km[i])
            result[i] = WaterVapourDensityToPressure(rho, T__kelvin)
    return result

def SlantPathAttenuation(f__ghz: np.ndarray, h_1__km: np.ndarray, h_2__km: np.ndarray, beta_1__rad: np.ndarray) -> SlantPathAttenuationResult:
    # Convert inputs to NumPy arrays if they're not already
    f__ghz = np.atleast_1d(f__ghz)
    h_1__km = np.atleast_1d(h_1__km)
    h_2__km = np.atleast_1d(h_2__km)
    beta_1__rad = np.atleast_1d(beta_1__rad)

    array_size = len(f__ghz)
    result = SlantPathAttenuationResult()

    result.A_gas__db = np.zeros(array_size)
    result.bending__rad = np.zeros(array_size)
    result.a__km = np.zeros(array_size)
    result.angle__rad = np.zeros(array_size)
    result.delta_L__km = np.zeros(array_size)

    for i in range(array_size):
        if beta_1__rad[i] > np.pi / 2:
            
            p__hPa = GlobalPressure(h_1__km[i])
            T__kelvin = GlobalTemperature(h_1__km[i])
            e__hPa = GlobalWetPressure(h_1__km[i])
            
            n_1 = RefractiveIndex(p__hPa, T__kelvin, e__hPa)

            h_G__km = h_1__km[i]
            delta = h_1__km[i] / 2
            diff = 100

            while abs(diff) > 0.001:
                if diff > 0:
                    h_G__km -= delta
                else:
                    h_G__km += delta
                delta /= 2

                p__hPa = GlobalPressure(h_G__km)
                T__kelvin = GlobalTemperature(h_G__km)
                e__hPa = GlobalWetPressure(h_G__km)
                
                n_G = RefractiveIndex(p__hPa, T__kelvin, e__hPa)

                grazing_term = n_G * (a_0__km + h_G__km)
                start_term = n_1 * (a_0__km + h_1__km[i]) * np.sin(beta_1__rad[i])

                diff = grazing_term - start_term
            
            beta_graze__rad = np.pi / 2
            result_1 = RayTrace(f__ghz[i], h_G__km, h_1__km[i], beta_graze__rad)
            result_2 = RayTrace(f__ghz[i], h_G__km, h_2__km[i], beta_graze__rad)

            result.angle__rad[i] = result_2.angle__rad
            result.A_gas__db[i] = result_1.A_gas__db + result_2.A_gas__db
            result.a__km[i] = result_1.a__km + result_2.a__km
            result.bending__rad[i] = result_1.bending__rad + result_2.bending__rad
            result.delta_L__km[i] = result_1.delta_L__km + result_2.delta_L__km
        else:
            single_result = RayTrace(f__ghz[i], h_1__km[i], h_2__km[i], beta_1__rad[i])
            
            result.angle__rad[i] = single_result.angle__rad.item()
            result.A_gas__db[i] = single_result.A_gas__db.item()
            result.a__km[i] = single_result.a__km.item()
            result.bending__rad[i] = single_result.bending__rad.item()
            result.delta_L__km[i] = single_result.delta_L__km.item()

    return result

def LayerThickness(m: np.ndarray, i: np.ndarray) -> np.ndarray:
    # Ensure inputs are NumPy arrays
    m = np.atleast_1d(m)
    i = np.atleast_1d(i)
    
    # Broadcast arrays to the same shape if necessary
    m, i = np.broadcast_arrays(m, i)
    
    # Equation 14
    delta_i__km = m * np.exp((i - 1) / 100.)
    
    return delta_i__km

def RayTrace(f__ghz, h_1__km, h_2__km, beta_1__rad) -> SlantPathAttenuationResult:
    result = SlantPathAttenuationResult()

    # Equations 16(a)-(c)
    i_lower = int(100 * np.log(1e4 * h_1__km * (np.exp(1. / 100.) - 1) + 1) + 1)
    i_upper = int(100 * np.log(1e4 * h_2__km * (np.exp(1. / 100.) - 1) + 1) + 1)
    m = ((np.exp(2. / 100.) - np.exp(1. / 100.)) / (np.exp(i_upper / 100.) - np.exp(i_lower / 100.))) * (h_2__km - h_1__km)

    # initialize starting layer
    delta_i__km = LayerThickness(m, i_lower)
    h_i__km = h_1__km + m * ((np.exp((i_lower - 1) / 100.) - np.exp((i_lower - 1) / 100.)) / (np.exp(1 / 100.) - 1))
    n_i, gamma_i = GetLayerProperties(f__ghz, h_i__km + delta_i__km / 2)
    r_i__km = a_0__km + h_i__km
    
    # record bottom layer properties for alpha and beta calculations
    r_1__km = r_i__km
    n_1 = n_i

    # summation from Equation 13
    for i in range(i_lower, i_upper):
        delta_ii__km = LayerThickness(m, i + 1)
        h_ii__km = h_1__km + m * ((np.exp(i / 100.) - np.exp((i_lower - 1) / 100.)) / (np.exp(1 / 100.) - 1))

        n_ii, gamma_ii = GetLayerProperties(f__ghz, h_ii__km + delta_ii__km / 2)

        r_ii__km = a_0__km + h_ii__km

        delta_i__km = LayerThickness(m, i)

        # Equation 19b
        beta_i__rad = np.arcsin(min(1, (n_1 * r_1__km) / (n_i * r_i__km) * np.sin(beta_1__rad)))

        # entry angle into the layer interface, Equation 18a
        alpha_i__rad = np.arcsin(min(1, (n_1 * r_1__km) / (n_i * r_ii__km) * np.sin(beta_1__rad)))

        # path length through ith layer, Equation 17
        a_i__km = -r_i__km * np.cos(beta_i__rad) + np.sqrt(r_i__km**2 * np.cos(beta_i__rad)**2 + 2 * r_i__km * delta_i__km + delta_i__km**2)
        
        result.a__km += a_i__km
        result.A_gas__db += a_i__km * gamma_i
        result.delta_L__km += a_i__km * (n_i - 1)     # summation, Equation 23

        beta_ii__rad = np.arcsin(n_i / n_ii * np.sin(alpha_i__rad))

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

def GlobalWetPressure(h__km: np.ndarray) -> np.ndarray:
    """
    Calculate the global wet pressure at given heights.

    Args:
    h__km (np.ndarray): Heights in kilometers

    Returns:
    np.ndarray: Wet pressures in hPa
    """
    T__kelvin = GlobalTemperature(h__km)
    P__hPa = GlobalPressure(h__km)
    rho__g_m3 = np.maximum(GlobalWaterVapourDensity(h__km, RHO_0__M_KG), 2 * 10**(-6) * 216.7 * P__hPa / T__kelvin)
    e__hPa = WaterVapourDensityToPressure(rho__g_m3, T__kelvin)

    return e__hPa

def RefractiveIndex(p__hPa: np.ndarray, T__kelvin: np.ndarray, e__hPa: np.ndarray) -> np.ndarray:
    """
    Calculate the refractive index based on pressure, temperature, and water vapor pressure.

    Args:
    p__hPa (np.ndarray): Pressures in hectopascals (hPa)
    T__kelvin (np.ndarray): Temperatures in Kelvin
    e__hPa (np.ndarray): Water vapor pressures in hectopascals (hPa)

    Returns:
    np.ndarray: Refractive indices
    """
    # dry term of refractivity
    N_dry = 77.6 * p__hPa / T__kelvin

    # wet term of refractivity
    N_wet = 72 * e__hPa / T__kelvin + 3.75e5 * e__hPa / np.power(T__kelvin, 2)

    N = N_dry + N_wet

    n = 1 + N * 10**(-6)

    return n

def SpecificAttenuation(f__ghz: np.ndarray, T__kelvin: np.ndarray, e__hPa: np.ndarray, p__hPa: np.ndarray) -> np.ndarray:
    """
    Calculate the specific attenuation due to atmospheric gases.

    Args:
    f__ghz (np.ndarray): Frequencies in GHz
    T__kelvin (np.ndarray): Temperatures in Kelvin
    e__hPa (np.ndarray): Water vapor pressures in hectopascals (hPa)
    p__hPa (np.ndarray): Total atmospheric pressures in hectopascals (hPa)

    Returns:
    np.ndarray: Specific attenuations in dB/km
    """
    gamma_o = OxygenSpecificAttenuation(f__ghz, T__kelvin, e__hPa, p__hPa)

    gamma_w = WaterVapourSpecificAttenuation(f__ghz, T__kelvin, e__hPa, p__hPa)

    gamma = gamma_o + gamma_w   # [Eqn 1]

    return gamma

OxygenData = {
    'f_0': np.array([
        50.474214,  50.987745,  51.503360,  52.021429,  52.542418,  53.066934,  53.595775,
        54.130025,  54.671180,  55.221384,  55.783815,  56.264774,  56.363399,  56.968211,
        57.612486,  58.323877,  58.446588,  59.164204,  59.590983,  60.306056,  60.434778,
        61.150562,  61.800158,  62.411220,  62.486253,  62.997984,  63.568526,  64.127775,
        64.678910,  65.224078,  65.764779,  66.302096,  66.836834,  67.369601,  67.900868,
        68.431006,  68.960312, 118.750334, 368.498246, 424.763020, 487.249273,
        715.392902, 773.839490, 834.145546
    ]),
    'a_1': np.array([
        0.975,    2.529,    6.193,   14.320,   31.240,   64.290,  124.600,  227.300,
        389.700,  627.100,  945.300,  543.400, 1331.800, 1746.600, 2120.100, 2363.700,
        1442.100, 2379.900, 2090.700, 2103.400, 2438.000, 2479.500, 2275.900, 1915.400,
        1503.000, 1490.200, 1078.000,  728.700,  461.300,  274.000,  153.000,   80.400,
        39.800,   18.560,    8.172,    3.397,    1.334,  940.300,   67.400,  637.700,
        237.400,   98.100,  572.300,  183.100
    ]),
    'a_2': np.array([
        9.651, 8.653, 7.709, 6.819, 5.983, 5.201, 4.474, 3.800, 3.182, 2.618, 2.109,
        0.014, 1.654, 1.255, 0.910, 0.621, 0.083, 0.387, 0.207, 0.207, 0.386, 0.621,
        0.910, 1.255, 0.083, 1.654, 2.108, 2.617, 3.181, 3.800, 4.473, 5.200, 5.982,
        6.818, 7.708, 8.652, 9.650, 0.010, 0.048, 0.044, 0.049, 0.145, 0.141, 0.145
    ]),
    'a_3': np.array([
        6.690,  7.170,  7.640,  8.110,  8.580,  9.060,  9.550,  9.960, 10.370,
        10.890, 11.340, 17.030, 11.890, 12.230, 12.620, 12.950, 14.910, 13.530,
        14.080, 14.150, 13.390, 12.920, 12.630, 12.170, 15.130, 11.740, 11.340,
        10.880, 10.380,  9.960,  9.550,  9.060,  8.580,  8.110,  7.640,  7.170,
        6.690, 16.640, 16.400, 16.400, 16.000, 16.000, 16.200, 14.700
    ]),
    'a_4': np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]),
    'a_5': np.array([
        2.566,  2.246,  1.947,  1.667,  1.388,  1.349,  2.227,  3.170,  3.558,  2.560,
        -1.172,  3.525, -2.378, -3.545, -5.416, -1.932,  6.768, -6.561,  6.957, -6.395,
        6.342,  1.014,  5.014,  3.029, -4.499,  1.856,  0.658, -3.036, -3.968, -3.528,
        -2.548, -1.660, -1.680, -1.956, -2.216, -2.492, -2.773, -0.439,  0.000,  0.000,
        0.000,  0.000,  0.000,  0.000
    ]),
    'a_6': np.array([
        6.850,  6.800,  6.729,  6.640,  6.526,  6.206,  5.085,  3.750,  2.654,  2.952,
        6.135, -0.978,  6.547,  6.451,  6.056,  0.436, -1.273,  2.309, -0.776,  0.699,
        -2.825, -0.584, -6.619, -6.759,  0.844, -6.675, -6.139, -2.895, -2.590, -3.680,
        -5.002, -6.091, -6.393, -6.475, -6.545, -6.600, -6.650,  0.079,  0.000,  0.000,
        0.000,  0.000,  0.000,  0.000
    ])
}

def OxygenRefractivity(f__ghz: np.ndarray, T__kelvin: np.ndarray, e__hPa: np.ndarray, p__hPa: np.ndarray) -> np.ndarray:
    """
    Calculate the imaginary part of the frequency-dependent complex refractivity due to oxygen.

    Args:
    f__ghz (np.ndarray): Frequencies in GHz
    T__kelvin (np.ndarray): Temperatures in Kelvin
    e__hPa (np.ndarray): Water vapour partial pressures in hectopascals (hPa)
    p__hPa (np.ndarray): Dry air pressures in hectopascals (hPa)

    Returns:
    np.ndarray: Refractivities in N-Units
    """
    # Ensure all inputs are NumPy arrays and broadcast to the same shape
    f__ghz, T__kelvin, e__hPa, p__hPa = np.broadcast_arrays(f__ghz, T__kelvin, e__hPa, p__hPa)
    
    theta = 300 / T__kelvin

    N = np.zeros_like(f__ghz)

    for i in range(len(OxygenData['f_0'])):
        # Equation 3, for oxygen
        S_i = OxygenData['a_1'][i] * 1e-7 * p__hPa * np.power(theta, 3) * np.exp(OxygenData['a_2'][i] * (1 - theta))

        # compute the width of the line, Equation 6a, for oxygen
        delta_f__ghz = OxygenData['a_3'][i] * 1e-4 * (p__hPa * np.power(theta, (0.8 - OxygenData['a_4'][i])) + 1.1 * e__hPa * theta)

        # modify the line width to account for Zeeman splitting of the oxygen lines
        # Equation 6b, for oxygen
        delta_f__ghz = np.sqrt(np.power(delta_f__ghz, 2) + 2.25e-6)

        # correction factor due to interference effects in oxygen lines
        # Equation 7, for oxygen
        delta = (OxygenData['a_5'][i] + OxygenData['a_6'][i] * theta) * 1e-4 * (p__hPa + e__hPa) * np.power(theta, 0.8)

        F_i = LineShapeFactor(f__ghz, OxygenData['f_0'][i], delta_f__ghz, delta)

        # summation of terms...from Equation 2a, for oxygen
        N += S_i * F_i

    N_D = NonresonantDebyeAttenuation(f__ghz, e__hPa, p__hPa, theta)

    N_o = N + N_D

    return N_o

def NonresonantDebyeAttenuation(f__ghz: np.ndarray, e__hPa: np.ndarray, p__hPa: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Calculate the Non-resonant Debye component of frequency-dependent complex refractivity.

    Args:
    f__ghz (np.ndarray): Frequencies in GHz
    e__hPa (np.ndarray): Water vapour partial pressures in hectopascals (hPa)
    p__hPa (np.ndarray): Dry air pressures in hectopascals (hPa)
    theta (np.ndarray): From Equation 3

    Returns:
    np.ndarray: Non-resonant Debye component
    """
    # Ensure all inputs are NumPy arrays and broadcast to the same shape
    f__ghz, e__hPa, p__hPa, theta = np.broadcast_arrays(f__ghz, e__hPa, p__hPa, theta)
    
    # width parameter for the Debye spectrum, Equation 9
    d = 5.6e-4 * (p__hPa + e__hPa) * np.power(theta, 0.8)

    # Equation 8
    frac_1 = 6.14e-5 / (d * (1 + np.power(f__ghz / d, 2)))
    frac_2 = (1.4e-12 * p__hPa * np.power(theta, 1.5)) / (1 + 1.9e-5 * np.power(f__ghz, 1.5))
    N_D = f__ghz * p__hPa * np.power(theta, 2) * (frac_1 + frac_2)

    return N_D

WaterVapourData = {
    'f_0': np.array([22.235080, 67.803960, 119.995940, 183.310087, 321.225630, 325.152888, 336.227764,
                     380.197353, 390.134508, 437.346667, 439.150807, 443.018343, 448.001085, 470.888999,
                     474.689092, 488.490108, 503.568532, 504.482692, 547.676440, 552.020960, 556.935985,
                     620.700807, 645.766085, 658.005280, 752.033113, 841.051732, 859.965698, 899.303175,
                     902.611085, 906.205957, 916.171582, 923.112692, 970.315022, 987.926764, 1780.000000]),
    'b_1': np.array([0.1079, 0.0011, 0.0007, 2.273, 0.0470, 1.514, 0.0010, 11.67, 0.0045,
                     0.0632, 0.9098, 0.1920, 10.41, 0.3254, 1.260, 0.2529, 0.0372, 0.0124,
                     0.9785, 0.1840, 497.0, 5.015, 0.0067, 0.2732, 243.4, 0.0134, 0.1325,
                     0.0547, 0.0386, 0.1836, 8.400, 0.0079, 9.009, 134.6, 17506.0]),
    'b_2': np.array([2.144, 8.732, 8.353, 0.668, 6.179, 1.541, 9.825, 1.048, 7.347, 5.048,
                     3.595, 5.048, 1.405, 3.597, 2.379, 2.852, 6.731, 6.731, 0.158, 0.158,
                     0.159, 2.391, 8.633, 7.816, 0.396, 8.177, 8.055, 7.914, 8.429, 5.110,
                     1.441, 10.293, 1.919, 0.257, 0.952]),
    'b_3': np.array([26.38, 28.58, 29.48, 29.06, 24.04, 28.23, 26.93, 28.11, 21.52, 18.45, 20.07,
                     15.55, 25.64, 21.34, 23.20, 25.86, 16.12, 16.12, 26.00, 26.00, 30.86, 24.38,
                     18.00, 32.10, 30.86, 15.90, 30.60, 29.85, 28.65, 24.08, 26.73, 29.00, 25.50,
                     29.85, 196.3]),
    'b_4': np.array([0.76, 0.69, 0.70, 0.77, 0.67, 0.64, 0.69, 0.54, 0.63, 0.60, 0.63, 0.60, 0.66, 0.66,
                     0.65, 0.69, 0.61, 0.61, 0.70, 0.70, 0.69, 0.71, 0.60, 0.69, 0.68, 0.33, 0.68, 0.68,
                     0.70, 0.70, 0.70, 0.70, 0.64, 0.68, 2.00]),
    'b_5': np.array([5.087, 4.930, 4.780, 5.022, 4.398, 4.893, 4.740, 5.063, 4.810, 4.230, 4.483,
                     5.083, 5.028, 4.506, 4.804, 5.201, 3.980, 4.010, 4.500, 4.500, 4.552, 4.856,
                     4.000, 4.140, 4.352, 5.760, 4.090, 4.530, 5.100, 4.700, 5.150, 5.000, 4.940,
                     4.550, 24.15]),
    'b_6': np.array([1.00, 0.82, 0.79, 0.85, 0.54, 0.74, 0.61, 0.89, 0.55, 0.48, 0.52, 0.50, 0.67, 0.65,
                     0.64, 0.72, 0.43, 0.45, 1.00, 1.00, 1.00, 0.68, 0.50, 1.00, 0.84, 0.45, 0.84,
                     0.90, 0.95, 0.53, 0.78, 0.80, 0.67, 0.90, 5.00])
}

def LineShapeFactor(f__ghz, f_i__ghz, delta_f__ghz, delta):
    term1 = f__ghz / f_i__ghz
    term2 = (delta_f__ghz - delta * (f_i__ghz - f__ghz)) / ((f_i__ghz - f__ghz)**2 + delta_f__ghz**2)
    term3 = (delta_f__ghz - delta * (f_i__ghz + f__ghz)) / ((f_i__ghz + f__ghz)**2 + delta_f__ghz**2)
    F_i = term1 * (term2 + term3)
    return F_i

def WaterVapourRefractivity(f__ghz: np.ndarray, T__kelvin: np.ndarray, e__hPa: np.ndarray, P__hPa: np.ndarray) -> np.ndarray:
    """
    Calculate the imaginary part of the frequency-dependent complex refractivity due to water vapour.

    Args:
    f__ghz (np.ndarray): Frequency in GHz
    T__kelvin (np.ndarray): Temperature in Kelvin
    e__hPa (np.ndarray): Water vapour partial pressure in hectopascals (hPa)
    P__hPa (np.ndarray): Dry air pressure in hectopascals (hPa)

    Returns:
    np.ndarray: Refractivity in N-Units
    """
    # Ensure all inputs are at least 1D arrays
    f__ghz = np.atleast_1d(f__ghz)
    T__kelvin = np.atleast_1d(T__kelvin)
    e__hPa = np.atleast_1d(e__hPa)
    P__hPa = np.atleast_1d(P__hPa)

    # Broadcast all inputs to the same shape
    f__ghz, T__kelvin, e__hPa, P__hPa = np.broadcast_arrays(f__ghz, T__kelvin, e__hPa, P__hPa)

    theta = 300 / T__kelvin
    N_w = np.zeros_like(f__ghz)
    
    for i in range(len(WaterVapourData['f_0'])):
        S_i = 0.1 * WaterVapourData['b_1'][i] * e__hPa * theta**3.5 * np.exp(WaterVapourData['b_2'][i] * (1 - theta))
        delta_f__ghz = 1e-4 * WaterVapourData['b_3'][i] * (P__hPa * theta**WaterVapourData['b_4'][i] + WaterVapourData['b_5'][i] * e__hPa * theta**WaterVapourData['b_6'][i])
        term1 = 0.217 * delta_f__ghz**2 + (2.1316e-12 * WaterVapourData['f_0'][i]**2 / theta)
        delta_f__ghz = 0.535 * delta_f__ghz + np.sqrt(term1)
        delta = np.zeros_like(f__ghz)
        F_i = LineShapeFactor(f__ghz, WaterVapourData['f_0'][i], delta_f__ghz, delta)
        N_w += S_i * F_i

    return N_w
    
def OxygenSpecificAttenuation(f__ghz: np.ndarray, T__kelvin: np.ndarray, e__hPa: np.ndarray, p__hPa: np.ndarray) -> np.ndarray:
    """
    Calculate the specific attenuation due to oxygen.

    Args:
    f__ghz (np.ndarray): Frequency in GHz
    T__kelvin (np.ndarray): Temperature in Kelvin
    e__hPa (np.ndarray): Water vapor partial pressure in hectopascals (hPa)
    p__hPa (np.ndarray): Dry air pressure in hectopascals (hPa)

    Returns:
    np.ndarray: Specific attenuation due to oxygen in dB/km
    """
    N_o = OxygenRefractivity(f__ghz, T__kelvin, e__hPa, p__hPa)
    gamma_o = 0.1820 * f__ghz * N_o
    return gamma_o

def WaterVapourSpecificAttenuation(f__ghz: np.ndarray, T__kelvin: np.ndarray, e__hPa: np.ndarray, p__hPa: np.ndarray) -> np.ndarray:
    """
    Calculate the specific attenuation due to water vapour.

    Args:
    f__ghz (np.ndarray): Frequency in GHz
    T__kelvin (np.ndarray): Temperature in Kelvin
    e__hPa (np.ndarray): Water vapor partial pressure in hectopascals (hPa)
    p__hPa (np.ndarray): Dry air pressure in hectopascals (hPa)

    Returns:
    np.ndarray: Specific attenuation due to water vapour in dB/km
    """
    N_w = WaterVapourRefractivity(f__ghz, T__kelvin, e__hPa, p__hPa)
    gamma_w = 0.1820 * f__ghz * N_w
    return gamma_w

def GetLayerProperties(f__ghz: np.ndarray, h_i__km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T__kelvin = GlobalTemperature(h_i__km)
    p__hPa = GlobalPressure(h_i__km)
    e__hPa = GlobalWetPressure(h_i__km)
    
    n = RefractiveIndex(p__hPa, T__kelvin, e__hPa)
    gamma = SpecificAttenuation(f__ghz, T__kelvin, e__hPa, p__hPa)

    return n, gamma

def ConvertToGeopotentialHeight(h__km: np.ndarray) -> np.ndarray:
    """
    Converts from geometric height, in km, to geopotential height, in km'.
    See Equation (1a).

    Args:
    h__km (np.ndarray): Geometric height, in km

    Returns:
    np.ndarray: Geopotential height, in km'
    """
    return (6356.766 * h__km) / (6356.766 + h__km)

def ConvertToGeometricHeight(h_prime__km):
    """
    Converts from geopotential height, in km', to geometric height, in km.
    See Equation (1b).

    Args:
    h_prime__km (np.ndarray): Geopotential height, in km'

    Returns:
    np.ndarray: Geometric height, in km
    """
    return (6356.766 * h_prime__km) / (6356.766 - h_prime__km)

def WaterVapourDensityToPressure(rho, T__kelvin):
    """
    Converts water vapour density, in g/m^3, to water vapour pressure, in hPa.
    See Equation (8).

    Args:
    rho (np.ndarray): Water vapour density, rho(h), in g/m^3
    T__kelvin (np.ndarray): Temperature, T(h), in Kelvin

    Returns:
    np.ndarray: Water vapour pressure, e(h), in hPa
    """
    return (rho * T__kelvin) / 216.7

def DistanceFunction(x__km):
    """
    Calculate the distance function G(x).

    Args:
    x__km (np.ndarray): Distance in km

    Returns:
    np.ndarray: Distance function value in dB
    """
    G_x__db = np.zeros_like(x__km)
    for i in range(len(x__km)):
        G_x__db[i] = 0.05751 * x__km[i] - 10.0 * np.log10(x__km[i])
    return G_x__db

def HeightFunction(x__km, K):
    """
    Calculate the height function F(x).

    Args:
    x__km (float or np.ndarray): Distance in km
    K (float or np.ndarray): Coefficient K

    Returns:
    float or np.ndarray: Height function value in dB
    """
    x__km = np.atleast_1d(x__km)
    K = np.atleast_1d(K)
    
    F_x__db = np.zeros_like(x__km)
    
    for i in range(len(x__km)):
        y__db = 40.0 * np.log10(x__km[i]) - 117.0
        G_x__db = DistanceFunction(np.array([x__km[i]]))[0]
        x_t__km = 450 / -(np.log10(K[i]) ** 3)  # Corrected line

        if x__km[i] <= 200.0:
            if x__km[i] >= x_t__km:
                if abs(y__db) < 117:
                    F_x__db[i] = y__db
                else:
                    F_x__db[i] = -117
            else:
                F_x__db[i] = 20 * np.log10(K[i]) - 15 + (0.000025 * x__km[i]**2 / K[i])
        elif x__km[i] > 2000.0:
            F_x__db[i] = G_x__db
        else:  # Blend y__db with G_x__db for 200 < x__km < 2000
            W = 0.0134 * x__km[i] * np.exp(-0.005 * x__km[i])
            F_x__db[i] = W * y__db + (1.0 - W) * G_x__db

    return F_x__db[0] if len(F_x__db) == 1 else F_x__db

def SmoothEarthDiffraction(d_1__km, d_2__km, f__mhz, d_0__km, T_pol):
    """
    Calculate the smooth earth diffraction loss.

    Args:
    d_1__km (np.ndarray): Horizon distance of terminal 1, in km
    d_2__km (np.ndarray): Horizon distance of terminal 2, in km
    f__mhz (np.ndarray): Frequency, in MHz
    d_0__km (np.ndarray): Path length of interest, in km
    T_pol (np.ndarray): Polarization code (0 for horizontal, 1 for vertical)

    Returns:
    np.ndarray: Diffraction loss in dB
    """
    THIRD = 1/3
    s = 18000 * sigma / f__mhz

    K = np.where(T_pol == POLARIZATION__HORIZONTAL,
                 0.01778 * f__mhz**(-THIRD) * ((epsilon_r - 1)**2 + s**2)**(-0.25),
                 0.01778 * f__mhz**(-THIRD) * ((epsilon_r**2 + s**2) / ((epsilon_r - 1)**2 + s**2)**0.5)**0.5)

    B_0 = 1.607

    x_0__km = (B_0 - K) * f__mhz**THIRD * d_0__km
    x_1__km = (B_0 - K) * f__mhz**THIRD * d_1__km
    x_2__km = (B_0 - K) * f__mhz**THIRD * d_2__km

    G_x__db = DistanceFunction(x_0__km)
    F_x1__db = HeightFunction(x_1__km, K)
    F_x2__db = HeightFunction(x_2__km, K)

    return G_x__db - F_x1__db - F_x2__db - 20.0

def FindPsiAtDistance(d__km: np.ndarray, path: Path, terminal_1: Terminal, terminal_2: Terminal) -> np.ndarray:
    # Ensure d__km is an array
    d__km = np.atleast_1d(d__km)
    
    # Initialize arrays
    psi = np.full_like(d__km, np.pi / 2)
    delta_psi = np.full_like(d__km, -np.pi / 4)
    
    # Process each instance individually
    for i in range(len(d__km)):
        if d__km[i] == 0:
            continue  # Skip the calculation for zero distance

        while True:
            psi[i] += delta_psi[i]

            # Create a LineOfSightParams object for this iteration
            params_temp = LineOfSightParams(
                z__km=np.zeros(2),
                d__km=np.zeros(1),
                r_0__km=np.zeros(1),
                r_12__km=np.zeros(1),
                D__km=np.zeros(2),
                theta_h1__rad=np.zeros(1),
                theta_h2__rad=np.zeros(1),
                theta=np.zeros(2),
                a_a__km=0.0,
                delta_r__km=np.zeros(1),
                A_LOS__db=np.zeros(1)
            )
            
            # Perform RayOptics for this instance
            RayOpticsVectorized(
                Terminal(**{k: np.array([v[i]]) if isinstance(v, np.ndarray) else v for k, v in vars(terminal_1).items()}),
                Terminal(**{k: np.array([v[i]]) if isinstance(v, np.ndarray) else v for k, v in vars(terminal_2).items()}),
                np.array([psi[i]]),
                params_temp
            )

            if params_temp.d__km[0] > d__km[i]:
                delta_psi[i] = abs(delta_psi[i]) / 2
            else:
                delta_psi[i] = -abs(delta_psi[i]) / 2

            if abs(d__km[i] - params_temp.d__km[0]) <= 1e-3 or abs(delta_psi[i]) <= 1e-12:
                break

    return psi[0] if len(psi) == 1 else psi

def FindPsiAtDeltaR(delta_r__km: np.ndarray, path: Path, terminal_1: Terminal, terminal_2: Terminal, terminate: np.ndarray) -> np.ndarray:
    # Initialize arrays
    psi = np.full_like(delta_r__km, np.pi / 2)
    delta_psi = np.full_like(delta_r__km, -np.pi / 4)
    
    # Process each instance individually
    for i in range(len(delta_r__km)):
        while True:
            psi[i] += delta_psi[i]

            # Create a LineOfSightParams object for this iteration
            params_temp = LineOfSightParams(
                z__km=np.zeros((1, 2)),
                d__km=np.zeros(1),
                r_0__km=np.zeros(1),
                r_12__km=np.zeros(1),
                D__km=np.zeros((1, 2)),
                theta_h1__rad=np.zeros(1),
                theta_h2__rad=np.zeros(1),
                theta=np.zeros((1, 2)),
                a_a__km=0.0,
                delta_r__km=np.zeros(1),
                A_LOS__db=np.zeros(1)
            )
            
            # Perform RayOptics for this instance
            RayOptics(
                Terminal(**{k: np.array([v[i]]) if isinstance(v, np.ndarray) else v for k, v in vars(terminal_1).items()}),
                Terminal(**{k: np.array([v[i]]) if isinstance(v, np.ndarray) else v for k, v in vars(terminal_2).items()}),
                psi[i],
                params_temp
            )

            if params_temp.delta_r__km[0] > delta_r__km[i]:
                delta_psi[i] = -abs(delta_psi[i]) / 2
            else:
                delta_psi[i] = abs(delta_psi[i]) / 2

            if abs(params_temp.delta_r__km[0] - delta_r__km[i]) <= terminate[i] or abs(delta_psi[i]) <= 1e-12:
                break

    return psi

def FindDistanceAtDeltaR(delta_r__km: np.ndarray, path: Path, terminal_1: Terminal, terminal_2: Terminal, terminate: np.ndarray) -> np.ndarray:
    # Initialize arrays
    psi = np.full_like(delta_r__km, np.pi / 2)
    delta_psi = np.full_like(delta_r__km, -np.pi / 4)
    d__km = np.zeros_like(delta_r__km)
    
    # Process each instance individually
    for i in range(len(delta_r__km)):
        while True:
            psi[i] += delta_psi[i]

            # Create a LineOfSightParams object for this iteration
            params_temp = LineOfSightParams(
                z__km=np.zeros((1, 2)),
                d__km=np.zeros(1),
                r_0__km=np.zeros(1),
                r_12__km=np.zeros(1),
                D__km=np.zeros((1, 2)),
                theta_h1__rad=np.zeros(1),
                theta_h2__rad=np.zeros(1),
                theta=np.zeros((1, 2)),
                a_a__km=0.0,
                delta_r__km=np.zeros(1),
                A_LOS__db=np.zeros(1)
            )
            
            # Perform RayOptics for this instance
            RayOptics(
                Terminal(**{k: np.array([v[i]]) if isinstance(v, np.ndarray) else v for k, v in vars(terminal_1).items()}),
                Terminal(**{k: np.array([v[i]]) if isinstance(v, np.ndarray) else v for k, v in vars(terminal_2).items()}),
                psi[i],
                params_temp
            )

            if params_temp.delta_r__km[0] > delta_r__km[i]:
                delta_psi[i] = -abs(delta_psi[i]) / 2
            else:
                delta_psi[i] = abs(delta_psi[i]) / 2

            if abs(params_temp.delta_r__km[0] - delta_r__km[i]) <= terminate[i]:
                d__km[i] = params_temp.d__km[0]
                break

    return d__km

def LineOfSight(path: Path, terminal_1: Terminal, terminal_2: Terminal, los_params: LineOfSightParams,
                f__mhz: np.ndarray, A_dML__db: np.ndarray, p: np.ndarray, d__km: np.ndarray, T_pol: np.ndarray, result: Result, K_LOS: np.ndarray) -> np.ndarray:

    # 0.2997925 = speed of light, gigameters per sec
    lambda__km = 0.2997925 / f__mhz  # [Eqn 6-1]
    terminate = lambda__km / 1e6

    # determine psi_limit, where you switch from free space to 2-ray model
    # lambda / 2 is the start of the lobe closest to d_ML
    psi_limit = FindPsiAtDeltaR(lambda__km / 2, path, terminal_1, terminal_2, terminate)
    
    # "[d_y6__km] is the largest distance at which a free-space value is obtained in a two-ray model
    #   of reflection from a smooth earth with a reflection coefficient of -1" [ES-83-3, page 44]
    d_y6__km = FindDistanceAtDeltaR(lambda__km / 6, path, terminal_1, terminal_2, terminate)
    
    # Determine d_0__km distance
    path.d_0__km = np.where(
        (terminal_1.d_r__km >= path.d_d__km) | (path.d_d__km >= path.d_ML__km),
        np.where(
            (terminal_1.d_r__km > d_y6__km) | (d_y6__km > path.d_ML__km),
            terminal_1.d_r__km,
            d_y6__km
        ),
        np.where(
            (path.d_d__km < d_y6__km) & (d_y6__km < path.d_ML__km),
            d_y6__km,
            path.d_d__km
        )
    )

    d_temp__km = path.d_0__km.copy()
    los_result = LineOfSightParams()
    
    while True:
        psi = FindPsiAtDistance(d_temp__km, path, terminal_1, terminal_2)
        los_result = RayOpticsVectorized(terminal_1, terminal_2, psi, los_result)
        
        #print(los_result)
        
        condition = (los_result.d__km[0] >= path.d_0__km) | ((d_temp__km + 0.001) >= path.d_ML__km)
        if np.all(condition):
            path.d_0__km = los_result.d__km[0]
            break
        
        d_temp__km += 0.001

    # Compute loss at d_0__km
    psi_d0 = FindPsiAtDistance(path.d_0__km, path, terminal_1, terminal_2)
    los_params = RayOpticsVectorized(terminal_1, terminal_2, np.array([psi_d0]), los_params)
    R_Tg = GetPathLoss(psi_d0, path, f__mhz, psi_limit, A_dML__db, np.zeros_like(A_dML__db), T_pol, los_params)
    
    
    # tune psi for the desired distance
    psi = FindPsiAtDistance(d__km, path, terminal_1, terminal_2)
    RayOpticsVectorized(terminal_1, terminal_2, psi, los_params)
    R_Tg = GetPathLoss(psi, path, f__mhz, psi_limit, A_dML__db, los_params.A_LOS__db, T_pol, los_params)

    print('PSI', psi, path, f__mhz, psi_limit, A_dML__db, los_params.A_LOS__db, T_pol, los_params)
    print('R_Tg', R_Tg)

    # Compute atmospheric absorption
    result_slant = SlantPathAttenuation(f__mhz / 1000, terminal_1.h_r__km, terminal_2.h_r__km, np.pi / 2 - los_params.theta_h1__rad)
    result.A_a__db = result_slant.A_gas__db

    # Compute free-space loss
    result.A_fs__db = 20.0 * np.log10(los_params.r_0__km) + 20.0 * np.log10(f__mhz) + 32.45  # [Eqn 6-4]

    print(result.A_fs__db)
    exit()

    # Compute variability
    f_theta_h = np.where(los_params.theta_h1__rad <= 0.0, 1.0,
                         np.where(los_params.theta_h1__rad >= 1.0, 0.0,
                                  np.maximum(0.5 - (1 / np.pi) * (np.arctan(20.0 * np.log10(32.0 * los_params.theta_h1__rad))), 0)))

    Y_e__db, A_Y = LongTermVariability(terminal_1.d_r__km, terminal_2.d_r__km, d__km, f__mhz, p, f_theta_h, los_params.A_LOS__db)
    Y_e_50__db, _ = LongTermVariability(terminal_1.d_r__km, terminal_2.d_r__km, d__km, f__mhz, 50 * np.ones_like(p), f_theta_h, los_params.A_LOS__db)

    F_AY = np.where(A_Y <= 0.0, 1.0,
                    np.where(A_Y >= 9.0, 0.1,
                             (1.1 + (0.9 * np.cos((A_Y / 9.0) * np.pi))) / 2.0))

    F_delta_r = np.where(los_params.delta_r__km >= (lambda__km / 2.0), 1.0,
                         np.where(los_params.delta_r__km <= lambda__km / 6.0, 0.1,
                                  0.5 * (1.1 - (0.9 * np.cos(((3.0 * np.pi) / lambda__km) * (los_params.delta_r__km - (lambda__km / 6.0)))))))

    R_s = R_Tg * F_delta_r * F_AY  # [Eqn 13-4]

    Y_pi_99__db = 10.0 * np.log10(f__mhz * np.power(result_slant.a__km, 3)) - 84.26  # [Eqn 13-5]
    K_t = FindKForYpiAt99Percent(Y_pi_99__db)

    W_a = np.power(10.0, K_t / 10.0)  # [Eqn 13-6]
    W_R = np.power(R_s, 2) + np.power(0.01, 2)  # [Eqn 13-7]
    W = W_R + W_a  # [Eqn 13-8]

    K_LOS = np.where(W <= 0.0, -40.0, np.maximum(10.0 * np.log10(W), -40.0))

    Y_pi_50__db = np.zeros_like(K_LOS)  # zero mean
    Y_pi__db = NakagamiRice(K_LOS, p)

    Y_total__db = -CombineDistributions(Y_e_50__db, Y_e__db, Y_pi_50__db, Y_pi__db, p)

    result.d__km = los_params.d__km
    result.A__db = result.A_fs__db + result.A_a__db - los_params.A_LOS__db + Y_total__db
    result.theta_h1__rad = los_params.theta_h1__rad

    return K_LOS

def RayOpticsVectorized(terminal_1: Terminal, terminal_2: Terminal, psi: np.ndarray, params: LineOfSightParams) -> LineOfSightParams:
    
    z = (a_0__km / a_e__km) - 1       # [Eqn 7-1]
    k_a = 1 / (1 + z * np.cos(psi))      # [Eqn 7-2]
    params.a_a__km = a_0__km * k_a          # [Eqn 7-3]

    delta_h_a1__km = terminal_1.delta_h__km * (params.a_a__km - a_0__km) / (a_e__km - a_0__km)  # [Eqn 7-4]
    delta_h_a2__km = terminal_2.delta_h__km * (params.a_a__km - a_0__km) / (a_e__km - a_0__km)  # [Eqn 7-4]
        
    H__km = np.array([terminal_1.h_r__km - delta_h_a1__km, terminal_2.h_r__km - delta_h_a2__km])    # [Eqn 7-5]
    
    params.z__km = params.a_a__km + H__km                                  # [Eqn 7-6]
    params.theta = np.arccos(params.a_a__km * np.cos(psi) / params.z__km) - psi   # [Eqn 7-7]
    params.D__km = params.z__km * np.sin(params.theta)                    # [Eqn 7-8]

    # [Eqn 7-9]
    Hprime__km = np.where(psi > 1.56, H__km, params.D__km * np.tan(psi))

    delta_z = np.abs(params.z__km[0] - params.z__km[1])   # [Eqn 7-10]

    params.d__km = np.maximum(params.a_a__km * (params.theta[0] + params.theta[1]), 0)  # [Eqn 7-11]

    alpha = np.where(
        (params.D__km[0] + params.D__km[1]) != 0,
        np.arctan((Hprime__km[1] - Hprime__km[0]) / (params.D__km[0] + params.D__km[1])),  # [Eqn 7-12]
        np.pi/2  # [Eqn 7-12]
    )
        
    params.r_0__km = np.maximum(delta_z, (params.D__km[0] + params.D__km[1]) / np.cos(alpha))            # [Eqn 7-13]
    params.r_12__km = (params.D__km[0] + params.D__km[1]) / np.cos(psi)                           # [Eqn 7-14]

    params.delta_r__km = 4.0 * Hprime__km[0] * Hprime__km[1] / (params.r_0__km + params.r_12__km)  # [Eqn 7-15]

    params.theta_h1__rad = alpha - params.theta[0]                # [Eqn 7-16]
    params.theta_h2__rad = -(alpha + params.theta[1])             # [Eqn 7-17]

    return params

def RayOptics(terminal_1: Terminal, terminal_2: Terminal, psi: float, params: LineOfSightParams) -> None:
    
    z = (a_0__km / a_e__km) - 1       # [Eqn 7-1]
    k_a = 1 / (1 + z * np.cos(psi))      # [Eqn 7-2]
    params.a_a__km = a_0__km * k_a          # [Eqn 7-3]

    delta_h_a1__km = terminal_1.delta_h__km * (params.a_a__km - a_0__km) / (a_e__km - a_0__km)  # [Eqn 7-4]
    delta_h_a2__km = terminal_2.delta_h__km * (params.a_a__km - a_0__km) / (a_e__km - a_0__km)  # [Eqn 7-4]
        
    H__km = [0, 0]
    H__km[0] = terminal_1.h_r__km - delta_h_a1__km    # [Eqn 7-5]
    H__km[1] = terminal_2.h_r__km - delta_h_a2__km    # [Eqn 7-5]
    
    Hprime__km = [0, 0]
    
    for i in range(2):
        #print(params.z__km[0, i])
        #exit()
        
        params.z__km[0, i] = params.a_a__km + H__km[i]                                  # [Eqn 7-6]
        params.theta[0, i] = np.arccos(params.a_a__km * np.cos(psi) / params.z__km[0, i]) - psi   # [Eqn 7-7]
        params.D__km[0, i] = params.z__km[0, i] * np.sin(params.theta[0, i])                    # [Eqn 7-8]

        # [Eqn 7-9]
        if psi > 1.56:
            Hprime__km[i] = H__km[i]
        else:
            Hprime__km[i] = params.D__km[0, i] * np.tan(psi)

    delta_z = abs(params.z__km[0, 0] - params.z__km[0, 1])   # [Eqn 7-10]

    params.d__km[0] = max(params.a_a__km * (params.theta[0, 0] + params.theta[0, 1]), 0)  # [Eqn 7-11]

    if (params.D__km[0, 0] + params.D__km[0, 1]) != 0:
        alpha = np.arctan((Hprime__km[1] - Hprime__km[0]) / (params.D__km[0, 0] + params.D__km[0, 1]))  # [Eqn 7-12]
    else:
        alpha = np.pi/2  # [Eqn 7-12]
        
    params.r_0__km[0] = max(delta_z, (params.D__km[0, 0] + params.D__km[0, 1]) / np.cos(alpha))            # [Eqn 7-13]
    params.r_12__km[0] = (params.D__km[0, 0] + params.D__km[0, 1]) / np.cos(psi)                           # [Eqn 7-14]

    params.delta_r__km[0] = 4.0 * Hprime__km[0] * Hprime__km[1] / (params.r_0__km[0] + params.r_12__km[0])  # [Eqn 7-15]

    params.theta_h1__rad[0] = alpha - params.theta[0, 0]                # [Eqn 7-16]
    params.theta_h2__rad[0] = -(alpha + params.theta[0, 1])             # [Eqn 7-17]

    return params

def GetPathLoss(psi__rad: np.ndarray, path: Path, f__mhz: np.ndarray, psi_limit: np.ndarray, 
                A_dML__db: np.ndarray, A_d_0__db: np.ndarray, T_pol: np.ndarray, 
                params: LineOfSightParams) -> np.ndarray:
    R_g, phi_g = ReflectionCoefficients(psi__rad, f__mhz, T_pol)

    D_v = np.where(np.tan(psi__rad) >= 0.1, 1.0, 
                   np.sqrt(1.0 + (2 * params.D__km[0] * params.D__km[1] * (1 + np.sin(psi__rad)**2)) / 
                           (params.a_a__km * params.r_12__km * np.sin(psi__rad)) + 
                           (2 * params.D__km[0] * params.D__km[1] / (params.a_a__km * params.r_12__km))**2)**-0.5)

    # Ray-length factor, [Eqn 8-6]
    F_r = np.minimum(params.r_0__km / params.r_12__km, 1)

    R_Tg = R_g * D_v * F_r  # [Eqn 8-7]

    # [Eqn 8-1]
    params.A_LOS__db = np.where(params.d__km > path.d_0__km,
                                ((params.d__km - path.d_0__km) * (A_dML__db - A_d_0__db) / (path.d_ML__km - path.d_0__km)) + A_d_0__db,
                                0)

    lambda__km = 0.2997925 / f__mhz  # [Eqn 8-2]

    # Total phase lag of the ground reflected ray relative to the direct ray
    phi_Tg = (2 * np.pi * params.delta_r__km / lambda__km) + phi_g  # [Eqn 8-8]

    # [Eqn 8-9]
    cplx = R_Tg * (np.cos(phi_Tg) - 1j * np.sin(phi_Tg))

    # [Eqn 8-10]
    W_RL = np.minimum(np.abs(1.0 + cplx), 1.0)

    # [Eqn 8-11]
    W_R0 = W_RL**2

    # [Eqn 8-12]
    A_LOS__db_psi = 10.0 * np.log10(W_R0)

    # Update A_LOS__db based on psi__rad and psi_limit
    params.A_LOS__db = np.where((params.d__km <= path.d_0__km) & (psi__rad <= psi_limit), A_LOS__db_psi, params.A_LOS__db)

    return R_Tg

def ReflectionCoefficients(psi__rad: np.ndarray, f__mhz: np.ndarray, T_pol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Ensure inputs are numpy arrays
    psi__rad = np.atleast_1d(psi__rad)
    f__mhz = np.atleast_1d(f__mhz)
    T_pol = np.atleast_1d(T_pol)

    # Handle boundary conditions
    psi__rad = np.clip(psi__rad, 0, np.pi/2)
    sin_psi = np.sin(psi__rad)
    cos_psi = np.cos(psi__rad)

    X = (18000.0 * sigma) / f__mhz              # [Eqn 9-1]
    Y = epsilon_r - cos_psi**2                  # [Eqn 9-2]
    T = np.sqrt(Y**2 + X**2) + Y                # [Eqn 9-3]
    P = np.sqrt(T * 0.5)                        # [Eqn 9-4]
    Q = X / (2.0 * P)                           # [Eqn 9-5]

    # [Eqn 9-6]
    B = np.where(T_pol == POLARIZATION__HORIZONTAL,
                 1.0 / (P**2 + Q**2),
                 (epsilon_r**2 + X**2) / (P**2 + Q**2))

    # [Eqn 9-7]
    A = np.where(T_pol == POLARIZATION__HORIZONTAL,
                 (2.0 * P) / (P**2 + Q**2),
                 (2.0 * (P * epsilon_r + Q * X)) / (P**2 + Q**2))

    # [Eqn 9-8]
    R_g = np.sqrt((1.0 + (B * sin_psi**2) - (A * sin_psi)) / (1.0 + (B * sin_psi**2) + (A * sin_psi)))

    # [Eqn 9-9]
    alpha = np.where(T_pol == POLARIZATION__HORIZONTAL,
                     np.arctan2(-Q, sin_psi - P),
                     np.arctan2((epsilon_r * sin_psi) - Q, epsilon_r * sin_psi - P))

    # [Eqn 9-10]
    beta = np.where(T_pol == POLARIZATION__HORIZONTAL,
                    np.arctan2(Q, sin_psi + P),
                    np.arctan2((X * sin_psi) + Q, epsilon_r * sin_psi + P))

    # [Eqn 9-11]
    phi_g = alpha - beta

    return R_g, phi_g


def LongTermVariability(d_r1__km: np.ndarray, d_r2__km: np.ndarray, d__km: np.ndarray, f__mhz: np.ndarray,
                        p: np.ndarray, f_theta_h: np.ndarray, A_T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the long term variability as described in Annex 2, Section 14 of
    Recommendation ITU-R P.528-5, "Propagation curves for aeronautical mobile
    and radionavigation services using the VHF, UHF and SHF bands"

    Args:
    d_r1__km (np.ndarray): Actual height of low terminal, in km
    d_r2__km (np.ndarray): Actual height of high terminal, in km
    d__km (np.ndarray): Path distance, in km
    f__mhz (np.ndarray): Frequency, in MHz
    p (np.ndarray): Time percentage
    f_theta_h (np.ndarray): Angular distance factor
    A_T (np.ndarray): Total loss

    Returns:
    tuple[np.ndarray, np.ndarray]: (Y_e__db, A_Y)
        Y_e__db: Variability, in dB
        A_Y: Conditional adjustment factor, in dB
    """
    THIRD = 1/3
    d_qs__km = 65.0 * np.power((100.0 / f__mhz), THIRD)  # [Eqn 14-1]
    d_Lq__km = d_r1__km + d_r2__km  # [Eqn 14-2]
    d_q__km = d_Lq__km + d_qs__km  # [Eqn 14-3]

    # [Eqn 14-4]
    d_e__km = np.where(d__km <= d_q__km,
                       (130.0 * d__km) / d_q__km,
                       130.0 + d__km - d_q__km)

    # [Eqns 14-5 and 14-6]
    g_10 = np.where(f__mhz > 1600.0,
                    1.05,
                    (0.21 * np.sin(5.22 * np.log10(f__mhz / 200.0))) + 1.28)
    g_90 = np.where(f__mhz > 1600.0,
                    1.05,
                    (0.18 * np.sin(5.22 * np.log10(f__mhz / 200.0))) + 1.23)

    c_1 = np.array([2.93e-4, 5.25e-4, 1.59e-5])
    c_2 = np.array([3.78e-8, 1.57e-6, 1.56e-11])
    c_3 = np.array([1.02e-7, 4.70e-7, 2.77e-8])

    n_1 = np.array([2.00, 1.97, 2.32])
    n_2 = np.array([2.88, 2.31, 4.08])
    n_3 = np.array([3.15, 2.90, 3.25])

    f_inf = np.array([3.2, 5.4, 0.0])
    f_m = np.array([8.2, 10.0, 3.9])

    Z__db = np.zeros((3, len(d_e__km)))
    for i in range(3):
        f_2 = f_inf[i] + ((f_m[i] - f_inf[i]) * np.exp(-c_2[i] * np.power(d_e__km, n_2[i])))
        Z__db[i] = (c_1[i] * np.power(d_e__km, n_1[i]) - f_2) * np.exp(-c_3[i] * np.power(d_e__km, n_3[i])) + f_2

    Y_p__db = np.where(p == 50, Z__db[2],
                       np.where(p > 50,
                                (special.erfcinv(2 * (1 - p/100)) / special.erfcinv(0.2)) * (-Z__db[0] * g_90) + Z__db[2],
                                np.where(p >= 10,
                                         (special.erfcinv(2 * (1 - p/100)) / special.erfcinv(1.8)) * (Z__db[1] * g_10) + Z__db[2],
                                         0)))  # placeholder for p < 10

    # Handle p < 10
    ps = np.array([1, 2, 5, 10])
    c_ps = np.array([1.9507, 1.7166, 1.3265, 1.0000])
    
    p_less_than_10 = p < 10
    if np.any(p_less_than_10):
        c_p = np.interp(p[p_less_than_10], ps, c_ps)
        Y_p__db[p_less_than_10] = c_p * (Z__db[1, p_less_than_10] * g_10[p_less_than_10]) + Z__db[2, p_less_than_10]

    Y_10__db = (Z__db[1] * g_10) + Z__db[2]  # [Eqn 14-20]
    Y_eI__db = f_theta_h * Y_p__db  # [Eqn 14-21]
    Y_eI_10__db = f_theta_h * Y_10__db  # [Eqn 14-22]

    A_YI = (A_T + Y_eI_10__db) - 3.0  # [Eqn 14-23]
    A_Y = np.maximum(A_YI, 0)  # [Eqn 14-24]
    Y_e__db = Y_eI__db - A_Y  # [Eqn 14-25]

    # For percentages less than 10%, do a correction check
    c_Y = np.array([-5.0, -4.5, -3.7, 0.0])
    P = np.array([1, 2, 5, 10])

    if np.any(p_less_than_10):
        c_Yi = np.interp(p[p_less_than_10], P, c_Y)
        Y_e__db[p_less_than_10] += A_T[p_less_than_10]
        Y_e__db[p_less_than_10] = np.minimum(Y_e__db[p_less_than_10], -c_Yi)
        Y_e__db[p_less_than_10] -= A_T[p_less_than_10]

    return Y_e__db, A_Y

def FindKForYpiAt99Percent(Y_pi_99__db: np.ndarray) -> np.ndarray:
    """
    Find K value for Y_pi at 99 percent.

    Args:
    Y_pi_99__db (np.ndarray): Y_pi values at 99 percent

    Returns:
    np.ndarray: Corresponding K values
    """
    # Ensure input is a numpy array
    Y_pi_99__db = np.atleast_1d(Y_pi_99__db)
    
    # Initialize output array
    K_out = np.zeros_like(Y_pi_99__db)
    
    for i, y in enumerate(Y_pi_99__db):
        # Is Y_pi_99__db smaller than the smallest value in the distribution data
        if y < NakagamiRiceCurves[0][Y_pi_99_INDEX]:
            K_out[i] = K[0]
            continue

        # Search the distribution data and interpolate to find K (dependent variable)
        for j in range(1, len(K)):
            if y - NakagamiRiceCurves[j][Y_pi_99_INDEX] < 0:
                K_out[i] = (K[j] * (y - NakagamiRiceCurves[j - 1][Y_pi_99_INDEX]) - 
                            K[j - 1] * (y - NakagamiRiceCurves[j][Y_pi_99_INDEX])) / \
                           (NakagamiRiceCurves[j][Y_pi_99_INDEX] - NakagamiRiceCurves[j - 1][Y_pi_99_INDEX])
                break
        else:
            # No match. Y_pi_99__db is greater than the data contains. Return largest K
            K_out[i] = K[-1]

    return K_out

# Data curves corresponding Nakagami-Rice distributions
NakagamiRiceCurves = [
    # K = -40 distribution
    [
        -0.1417, -0.1252, -0.1004, -0.0784, -0.0634,
        -0.0515, -0.0321, -0.0155,  0.0000,  0.0156,  0.0323,
         0.0518,  0.0639,  0.0791,  0.1016,  0.1271,  0.1441
    ],
    [
        -0.7676, -0.6811, -0.5497, -0.4312, -0.3504,
        -0.2856, -0.1790, -0.0870,  0.0000,  0.0878,  0.1828,
         0.2953,  0.3651,  0.4537,  0.5868,  0.7390,  0.8420
    ],
    [
        -1.3183, -1.1738, -0.9524, -0.7508, -0.6121,
        -0.5003, -0.3151, -0.1537,  0.0000,  0.1564,  0.3269,
         0.5308,  0.6585,  0.8218,  1.0696,  1.3572,  1.5544
    ],
    [
        -1.6263, -1.4507, -1.1805, -0.9332, -0.7623,
        -0.6240, -0.3940, -0.1926,  0.0000,  0.1969,  0.4127,
         0.6722,  0.8355,  1.0453,  1.3660,  1.7417,  2.0014
    ],
    [
        -1.9963, -1.7847, -1.4573, -1.1557, -0.9462,
        -0.7760, -0.4916, -0.2410,  0.0000,  0.2478,  0.5209,
         0.8519,  1.0615,  1.3326,  1.7506,  2.2463,  2.5931
    ],
    [
        -2.4355, -2.1829, -1.7896, -1.4247, -1.1695,
        -0.9613, -0.6113, -0.3007,  0.0000,  0.3114,  0.6573,
         1.0802,  1.3505,  1.7028,  2.2526,  2.9156,  3.3872
    ],
    [
        -2.9491, -2.6507, -2.1831, -1.7455, -1.4375,
        -1.1846, -0.7567, -0.3737,  0.0000,  0.3903,  0.8281,
         1.3698,  1.7198,  2.1808,  2.9119,  3.8143,  4.4714
    ],
    [
        -3.5384, -3.1902, -2.6407, -2.1218, -1.7535,
        -1.4495, -0.9307, -0.4619,  0.0000,  0.4874,  1.0404,
         1.7348,  2.1898,  2.7975,  3.7820,  5.0373,  5.9833
    ],
    [
        -4.1980, -3.7974, -3.1602, -2.5528, -2.1180,
        -1.7565, -1.1345, -0.5662,  0.0000,  0.6045,  1.2999,
         2.1887,  2.7814,  3.5868,  4.9288,  6.7171,  8.1319
    ],
    [
        -4.9132, -4.4591, -3.7313, -3.0306, -2.5247,
        -2.1011, -1.3655, -0.6855,  0.0000,  0.7415,  1.6078,
         2.7374,  3.5059,  4.5714,  6.4060,  8.9732, 11.0973
    ],
    [
        -5.6559, -5.1494, -4.3315, -3.5366, -2.9578,
        -2.4699, -1.6150, -0.8154,  0.0000,  0.8935,  1.9530,
         3.3611,  4.3363,  5.7101,  8.1216, 11.5185, 14.2546
    ],
    [
        -6.3810, -5.8252, -4.9219, -4.0366, -3.3871,
        -2.8364, -1.8638, -0.9455,  0.0000,  1.0458,  2.2979,
         3.9771,  5.1450,  6.7874,  9.6276, 13.4690, 16.4251
    ],
    [
        -7.0247, -6.4249, -5.4449, -4.4782, -3.7652,
        -3.1580, -2.0804, -1.0574,  0.0000,  1.1723,  2.5755,
         4.4471,  5.7363,  7.5266, 10.5553, 14.5401, 17.5511
    ],
    [
        -7.5229, -6.8862, -5.8424, -4.8090, -4.0446,
        -3.3927, -2.2344, -1.1347,  0.0000,  1.2535,  2.7446,
         4.7144,  6.0581,  7.9073, 11.0003, 15.0270, 18.0526
    ],
    [
        -7.8532, -7.1880, -6.0963, -5.0145, -4.2145,
        -3.5325, -2.3227, -1.1774,  0.0000,  1.2948,  2.8268,
         4.8377,  6.2021,  8.0724, 11.1869, 15.2265, 18.2566
    ],
    [
        -8.0435, -7.3588, -6.2354, -5.1234, -4.3022,
        -3.6032, -2.3656, -1.1975,  0.0000,  1.3130,  2.8619,
         4.8888,  6.2610,  8.1388, 11.2607, 15.3047, 18.3361
    ],
    [
        -8.2238, -7.5154, -6.3565, -5.2137, -4.3726,
        -3.6584, -2.3979, -1.2121,  0.0000,  1.3255,  2.8855,
         4.9224,  6.2992,  8.1814, 11.3076, 15.3541, 18.3864
    ]
]

K = [
    -40, -25, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 20
]

# Percentages for interpolation and data tables
P = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 85, 90, 95, 98, 99]

def linear_interpolation(x, x1, y1, x2, y2):
    return y1 + (x - x1) * (y2 - y1) / (x2 - x1)

def NakagamiRice(K_value: np.ndarray, p_value: np.ndarray) -> np.ndarray:
    # Ensure inputs are numpy arrays
    K_value = np.atleast_1d(K_value)
    p_value = np.atleast_1d(p_value)
    
    # Find the indices for K and P
    d_K = np.searchsorted(K, K_value, side='right')
    d_p = np.searchsorted(P, p_value, side='right')
    
    # Initialize output array
    result = np.zeros(K_value.shape)
    
    for i in range(len(K_value)):
        if d_K[i] == 0:  # K_value <= smallest K
            if d_p[i] == 0:
                result[i] = NakagamiRiceCurves[0][0]
            else:
                result[i] = linear_interpolation(
                    p_value[i], P[d_p[i] - 1], NakagamiRiceCurves[0][d_p[i] - 1],
                    P[d_p[i]], NakagamiRiceCurves[0][d_p[i]]
                )
        elif d_K[i] == len(K):  # K_value > largest K
            if d_p[i] == 0:
                result[i] = NakagamiRiceCurves[-1][0]
            else:
                result[i] = linear_interpolation(
                    p_value[i], P[d_p[i] - 1], NakagamiRiceCurves[-1][d_p[i] - 1],
                    P[d_p[i]], NakagamiRiceCurves[-1][d_p[i]]
                )
        else:
            if d_p[i] == 0:
                result[i] = linear_interpolation(
                    K_value[i], K[d_K[i] - 1], NakagamiRiceCurves[d_K[i] - 1][0],
                    K[d_K[i]], NakagamiRiceCurves[d_K[i]][0]
                )
            else:
                v1 = linear_interpolation(
                    K_value[i], K[d_K[i] - 1], NakagamiRiceCurves[d_K[i] - 1][d_p[i]],
                    K[d_K[i]], NakagamiRiceCurves[d_K[i]][d_p[i]]
                )
                v2 = linear_interpolation(
                    K_value[i], K[d_K[i] - 1], NakagamiRiceCurves[d_K[i] - 1][d_p[i] - 1],
                    K[d_K[i]], NakagamiRiceCurves[d_K[i]][d_p[i] - 1]
                )
                result[i] = linear_interpolation(p_value[i], P[d_p[i] - 1], v2, P[d_p[i]], v1)
    
    return result

def CombineDistributions(A_M: np.ndarray, A_p: np.ndarray, B_M: np.ndarray, B_p: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Combine distributions with vectorized inputs.

    Args:
    A_M, A_p, B_M, B_p, p (np.ndarray): Input arrays

    Returns:
    np.ndarray: Combined distribution values
    """
    C_M = A_M + B_M

    Y_1 = A_p - A_M
    Y_2 = B_p - B_M

    Y_3 = np.sqrt(Y_1**2 + Y_2**2)

    return np.where(p < 50, C_M + Y_3, C_M - Y_3)

def InverseComplementaryCumulativeDistributionFunction(q: np.ndarray) -> np.ndarray:
    """
    Compute the inverse complementary cumulative distribution function with vectorized input.

    Args:
    q (np.ndarray): Input array

    Returns:
    np.ndarray: Computed Q_q values
    """
    C_0 = 2.515516
    C_1 = 0.802853
    C_2 = 0.010328
    D_1 = 1.432788
    D_2 = 0.189269
    D_3 = 0.001308

    x = np.where(q > 0.5, 1.0 - q, q)

    T_x = np.sqrt(-2.0 * np.log(x))

    zeta_x = ((C_2 * T_x + C_1) * T_x + C_0) / (((D_3 * T_x + D_2) * T_x + D_1) * T_x + 1.0)

    Q_q = T_x - zeta_x

    return np.where(q > 0.5, -Q_q, Q_q)