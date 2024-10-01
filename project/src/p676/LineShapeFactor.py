from p676 import *

def LineShapeFactor(f__ghz, f_i__ghz, delta_f__ghz, delta):
    term1 = f__ghz / f_i__ghz
    term2 = (delta_f__ghz - delta * (f_i__ghz - f__ghz)) / ((f_i__ghz - f__ghz)**2 + delta_f__ghz**2)
    term3 = (delta_f__ghz - delta * (f_i__ghz + f__ghz)) / ((f_i__ghz + f__ghz)**2 + delta_f__ghz**2)

    F_i = term1 * (term2 + term3)

    return F_i