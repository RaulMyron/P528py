from dataclasses import dataclass, field
from typing import List

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
class Data:
    P: List[float] = []  # Percentages for interpolation and data tables
    NakagamiRiceCurves: List[List[float]] = []
    K: List[int] = []

@dataclass
class Path:
    d_ML__km: float = 0.0
    d_0__km: float = 0.0
    d_d__km: float = 0.0

@dataclass
class Terminal:
    h_r__km: float = 0.0
    h_e__km: float = 0.0
    delta_h__km: float = 0.0
    d_r__km: float = 0.0
    a__km: float = 0.0
    phi__rad: float = 0.0
    theta__rad: float = 0.0
    A_a__db: float = 0.0

@dataclass
class LineOfSightParams:
    z__km: List[float] = field(default_factory=lambda: [0.0, 0.0])
    d__km: float = 0.0
    r_0__km: float = 0.0
    r_12__km: float = 0.0
    D__km: List[float] = field(default_factory=lambda: [0.0, 0.0])
    theta_h1__rad: float = 0.0
    theta_h2__rad: float = 0.0
    theta: List[float] = field(default_factory=lambda: [0.0, 0.0])
    a_a__km: float = 0.0
    delta_r__km: float = 0.0
    A_LOS__db: float = 0.0

@dataclass
class TroposcatterParams:
    d_s__km: float = 0.0
    d_z__km: float = 0.0
    h_v__km: float = 0.0
    theta_s: float = 0.0
    theta_A: float = 0.0
    A_s__db: float = 0.0
    A_s_prev__db: float = 0.0
    M_s: float = 0.0

@dataclass
class Result:
    propagation_mode: int = 0
    d__km: float = 0.0
    A__db: float = 0.0
    A_fs__db: float = 0.0
    A_a__db: float = 0.0
    theta_h1__rad: float = 0.0
    result: str = ''