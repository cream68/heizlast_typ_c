import pytest
from heizlast_typ_c.interpolation import interpolate_bilinear_with_fallback, interpolate_linear_with_fallback, get_bounding_box, get_bounding_box_1d
from heizlast_typ_c.const import overcoverage_factors, pipe_spacing_factors

row_upper_0075 = 1.046+(1.035-1.046)/(0.1-0.05)*(0.075-0.05)
row_lower_0075 = 1.041+(1.0315-1.041)/(0.1-0.05)*(0.075-0.05)
value = row_lower_0075+(row_upper_0075-row_lower_0075)/(0.2-0.15)*(0.175-0.15)

@pytest.mark.parametrize(
    "T, R_lambda_B, expected",
    [
        (0.05, 0.0, 1.069),                  # Exact upper left corner
        (0.375, 0.15, 1.015),                # Exact lower right corner
        (0.375, 0.15, 1.015),                # Exact lower right corner
        (0.05,0.025,(1.069+1.056)/2),
        (0.375,0.10,1.0181),
        (0.375,0.125,(1.0181+1.015)/2),
        (0.175,0.075,value), 
        (0.3,0.15,1.021),
        (0.3,0.0,1.0395)                     
    ]
)
def test_bilinear_interpolation(T, R_lambda_B, expected):
    T1, T2, R1, R2, Q11, Q12, Q21, Q22 = get_bounding_box(T, R_lambda_B, overcoverage_factors)
    latexcode, result = interpolate_bilinear_with_fallback(
    T=T, R=R_lambda_B,
    T1=T1, T2=T2,
    R1=R1, R2=R2,
    Q11=Q11, Q12=Q12,
    Q21=Q21, Q22=Q22
)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "T, expected",
    [
        (0.0, 1.23),                  # Exact upper left corner
        (0.15, 1.134),                # Exact lower right corner
        (0.4, 1.134),                # Exact lower right corner
        (0.075, 0.5*(1.188+1.156)),                # Exact lower right corner
    ]
)
def test_linear_interpolation(T, expected):
    T1, T2, Q1, Q2 = get_bounding_box_1d(T, pipe_spacing_factors)
    latexcode, result = interpolate_linear_with_fallback(
    T=T,
    T1=T1, T2=T2,
    Q1=Q1, Q2=Q2
)
    assert abs(result - expected) < 1e-10, f"Expected {expected}, got {result}"
