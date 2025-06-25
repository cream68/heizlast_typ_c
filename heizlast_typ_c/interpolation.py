import numpy as np
from handcalcs.decorator import handcalc
import math

EPSILON = 1e-6

def extract_sorted_axes(data):
    T_vals = sorted(set(k[0] for k in data))
    R_vals = sorted(set(k[1] for k in data))
    return T_vals, R_vals

from handcalcs.decorator import handcalc

@handcalc(precision=4)
def bilinear_1d_r(R, R1, R2, Q11, Q12):
    f = Q11 + (Q12 - Q11) * (R - R1) / (R2 - R1)
    return f

@handcalc(precision=4)
def bilinear_1d_t(T, T1, T2, Q11, Q21):
    f = Q11 + (Q21 - Q11) * (T - T1) / (T2 - T1)
    return f

@handcalc(precision=4)
def linear_1d(T, T1, T2, Q1, Q2):
    f = Q1 + (Q2 - Q1) * (T - T1) / (T2 - T1)
    return f

@handcalc(precision=4)
def bilinear_2d(T, R, T1, T2, R1, R2, Q11, Q12, Q21, Q22):
    f_clower = Q11 + (Q21 - Q11) * (T - T1) / (T2 - T1)
    f_cupper = Q12 + (Q22 - Q12) * (T - T1) / (T2 - T1)
    f = f_clower + (f_cupper - f_clower) * (R - R1) / (R2 - R1)
    return f

@handcalc(precision=4)
def exact(Q11):
    f = Q11
    return f

def interpolate_bilinear_with_fallback(T, R, T1, T2, R1, R2, Q11, Q12, Q21, Q22):
    # Degenerate cases: boundaries collapsed (clamping)
    if math.isclose(T1, T2, abs_tol=EPSILON) and math.isclose(R1, R2, abs_tol=EPSILON):
        return exact(Q11)  # or any of Q11–Q22; all are same
    elif math.isclose(T1, T2, abs_tol=EPSILON):
        return interpolate_linear_with_fallback(R, R1, R2, Q11, Q12)
    elif math.isclose(R1, R2, abs_tol=EPSILON):
        return interpolate_linear_with_fallback(T, T1, T2, Q11, Q21)

    # Exact node matches
    if math.isclose(T, T1, abs_tol=EPSILON) and math.isclose(R, R1, abs_tol=EPSILON):
        return exact(Q11)
    elif math.isclose(T, T1, abs_tol=EPSILON) and math.isclose(R, R2, abs_tol=EPSILON):
        return exact(Q12)
    elif math.isclose(T, T2, abs_tol=EPSILON) and math.isclose(R, R1, abs_tol=EPSILON):
        return exact(Q21)
    elif math.isclose(T, T2, abs_tol=EPSILON) and math.isclose(R, R2, abs_tol=EPSILON):
        return exact(Q22)

    # One dimension exact → use linear interpolation
    elif math.isclose(T, T1, abs_tol=EPSILON):
        return interpolate_linear_with_fallback(R, R1, R2, Q11, Q12)
    elif math.isclose(T, T2, abs_tol=EPSILON):
        return interpolate_linear_with_fallback(R, R1, R2, Q21, Q22)
    elif math.isclose(R, R1, abs_tol=EPSILON):
        return interpolate_linear_with_fallback(T, T1, T2, Q11, Q21)
    elif math.isclose(R, R2, abs_tol=EPSILON):
        return interpolate_linear_with_fallback(T, T1, T2, Q12, Q22)

    # Full bilinear interpolation
    return bilinear_2d(T, R, T1, T2, R1, R2, Q11, Q12, Q21, Q22)
    
def interpolate_linear_with_fallback(T, T1, T2, Q1, Q2):
    if math.isclose(T1, T2, abs_tol=EPSILON):
        return exact(Q1)  # or Q2, since Q1 == Q2 in this context
    if math.isclose(T, T1, abs_tol=EPSILON):
        return exact(Q1)
    if math.isclose(T, T2, abs_tol=EPSILON):
        return exact(Q2)
    else: 
        return linear_1d(T, T1, T2, Q1, Q2)

def get_bounding_box(T, R_lambda_B, factors_dict):
    T_vals = sorted(set(t for (t, r) in factors_dict))
    R_vals = sorted(set(r for (t, r) in factors_dict))

    # Clamp or interpolate T
    if T <= T_vals[0]:
        T1 = T2 = T_vals[0]
    elif T >= T_vals[-1]:
        T1 = T2 = T_vals[-1]
    else:
        T1 = max(t for t in T_vals if t <= T)
        T2 = min(t for t in T_vals if t >= T and t != T1)

    # Clamp or interpolate R
    if R_lambda_B <= R_vals[0]:
        R1 = R2 = R_vals[0]
    elif R_lambda_B >= R_vals[-1]:
        R1 = R2 = R_vals[-1]
    else:
        R1 = max(r for r in R_vals if r <= R_lambda_B)
        R2 = min(r for r in R_vals if r >= R_lambda_B and r != R1)

    # Extract corresponding Q-values from the dictionary
    Q11 = factors_dict.get((T1, R1))
    Q12 = factors_dict.get((T1, R2))
    Q21 = factors_dict.get((T2, R1))
    Q22 = factors_dict.get((T2, R2))

    return T1, T2, R1, R2, Q11, Q12, Q21, Q22


def get_bounding_box_1d(x, factors_dict):
    """
    Returns the bounding keys and values for 1D interpolation from a dict.

    Parameters:
    - x: The query value.
    - factors_dict: Dictionary with scalar keys and corresponding values.

    Returns:
    - x1, x2: Bounding keys (can be equal if x is outside range).
    - y1, y2: Corresponding values from the dictionary.
    """
    keys = sorted(factors_dict.keys())

    # Clamp left
    if x <= keys[0]:
        x1 = x2 = keys[0]
    # Clamp right
    elif x >= keys[-1]:
        x1 = x2 = keys[-1]
    # Interpolation range
    else:
        x1 = max(k for k in keys if k <= x)
        x2 = min(k for k in keys if k >= x and k != x1)

    y1 = factors_dict[x1]
    y2 = factors_dict[x2]

    return x1, x2, y1, y2
