from typing import List, Tuple

# Constants
PI = 3.1415926535897932384
a_0__km = 6371.0
a_e__km = 9257.0
N_s = 341
epsilon_r = 15.0
sigma = 0.005
LOS_EPSILON = 0.00001
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

# RETURN CODES
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
class Data:
    P: List[float] = []
    NakagamiRiceCurves: List[List[float]] = []
    K: List[int] = []

# Data Structures
class Path:
    def __init__(self):
        self.d_ML__km: float = 0.0
        self.d_0__km: float = 0.0
        self.d_d__km: float = 0.0

class Terminal:
    def __init__(self):
        self.h_r__km: float = 0.0
        self.h_e__km: float = 0.0
        self.delta_h__km: float = 0.0
        self.d_r__km: float = 0.0
        self.a__km: float = 0.0
        self.phi__rad: float = 0.0
        self.theta__rad: float = 0.0
        self.A_a__db: float = 0.0

class LineOfSightParams:
    def __init__(self):
        self.z__km: Tuple[float, float] = (0.0, 0.0)
        self.d__km: float = 0.0
        self.r_0__km: float = 0.0
        self.r_12__km: float = 0.0
        self.D__km: Tuple[float, float] = (0.0, 0.0)
        self.theta_h1__rad: float = 0.0
        self.theta_h2__rad: float = 0.0
        self.theta: Tuple[float, float] = (0.0, 0.0)
        self.a_a__km: float = 0.0
        self.delta_r__km: float = 0.0
        self.A_LOS__db: float = 0.0

class TroposcatterParams:
    def __init__(self):
        self.d_s__km: float = 0.0
        self.d_z__km: float = 0.0
        self.h_v__km: float = 0.0
        self.theta_s: float = 0.0
        self.theta_A: float = 0.0
        self.A_s__db: float = 0.0
        self.A_s_prev__db: float = 0.0
        self.M_s: float = 0.0

class Result:
    def __init__(self):
        self.propagation_mode: int = 0
        self.d__km: float = 0.0
        self.A__db: float = 0.0
        self.A_fs__db: float = 0.0
        self.A_a__db: float = 0.0
        self.theta_h1__rad: float = 0.0
