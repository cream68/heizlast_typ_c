import streamlit as st
import numpy as np
from scipy.optimize import fsolve
from pint import UnitRegistry
from handcalcs.decorator import handcalc
from heizlast_typ_c.interpolation import get_bounding_box, get_bounding_box_1d, interpolate_linear_with_fallback, interpolate_bilinear_with_fallback
from heizlast_typ_c.const import c_water, overcoverage_factors, pipe_diameter_factors, pipe_spacing_factors, water_density
import handcalcs

handcalcs.set_option("preferred_string_formatter", "~L")
ureg = UnitRegistry()
ureg.auto_reduce_dimensions = True

# --- Helper Functions ---

def berechne_massestrom(Q, T_VL, T_RL, c_water):
    delta_T = T_VL - T_RL
    return (Q / (c_water * delta_T)) * 3600

def berechne_ruecklauf(delta_t_log, theta_VL, theta_room):
    def equation(theta_RL):
        return (theta_VL - theta_RL) / np.log((theta_VL - theta_room) / (theta_RL - theta_room)) - delta_t_log
    return fsolve(equation, theta_room)[0]

def calculate_specific_heat_output(B0, ab, aT, aD, au, mt, mu, mD):
    return B0 * ab * aT**mt * aD**mD * au**mu

# --- LaTeX-based formulas with handcalc ---

@handcalc()
def formula_ab(alpha, s_u, R_lambda_B, lambdau0, lambdaE):
    a_b = (1 / alpha + s_u / lambdau0) / (1 / alpha + s_u / lambdaE + R_lambda_B)
    return a_b

@handcalc()
def formula_mt(T):
    m_t = 1 - T / 0.075
    return locals()

@handcalc()
def formula_mu(s_u):
    m_u = 100 * (0.045 - s_u)
    return locals()

@handcalc()
def formula_md(D):
    m_D = 250 * (D - 0.02)
    return locals()

# --- UI Start ---

st.set_page_config(page_title="Heizleistung Typ C")
st.title("🔧 Heizleistung Typ C")

with st.sidebar:
    st.header("📥 Eingabewerte")
    col1, col2 = st.columns(2)
    with col1:
        B0 = st.number_input("Wärmeleitfähigkeit Rohr (W/m²·K)", value=6.7,help="B0 = 6,7 W/(m2 ∙   K)  für  eine  Wärmeleitfähigkeit  des  Rohres λR = λR,0 = 0,35 W/m2K  und  eine  Rohrwanddicke sR = sR,0 = (da – di)/2 = 0,002 m")
        T = st.number_input("Rohrabstand T (m)", value=0.2, format="%0.3f")
        D = st.number_input("Rohraußendurchmesser D (m)", value=0.016, format="%0.3f")
        su = st.number_input("Überdeckung sᵤ (m)", value=0.045, format="%0.3f")
        R_lambda_B = st.number_input("Wärmeleitwiderstand des Bodenbelags (m²·K/W)", value=0.015, min_value=0.000, max_value=0.150, format="%0.3f")
    with col2:
        room_size = st.number_input("Raumfläche (m²)", value=28.79)
        room_heatload = st.number_input("Raum-Heizlast (W)", value=400.0)
        theta_room = st.number_input("Raumtemperatur (°C)", value=24.0)
        theta_VL = st.number_input("Vorlauftemperatur (°C)", value=35.0)

# Constants
lambda_E = 1.2
lambdau0 = 1.0

# Calculate a_b
latex_ab, a_b = formula_ab(alpha=10.8, s_u=su, R_lambda_B=R_lambda_B, lambdau0=lambdau0, lambdaE=lambda_E)

# Get m-factors
latex_mt, result_mt = formula_mt(T)
m_t = result_mt["m_t"]

latex_mu, result_mu = formula_mu(su)
m_u = result_mu["m_u"]

latex_md, result_md = formula_md(D)
m_D = result_md["m_D"]

# Interpolated aT, au, aD
T1, T2, Q1, Q2 = get_bounding_box_1d(R_lambda_B, pipe_spacing_factors)
latex_aT, aT = interpolate_linear_with_fallback(T=R_lambda_B, T1=T1, T2=T2, Q1=Q1, Q2=Q2)
T1, T2, R1, R2, Q11, Q12, Q21, Q22 = get_bounding_box(T, R_lambda_B, overcoverage_factors)
latex_au, au = interpolate_bilinear_with_fallback(
    T=T, R=R_lambda_B,
    T1=T1, T2=T2,
    R1=R1, R2=R2,
    Q11=Q11, Q12=Q12,
    Q21=Q21, Q22=Q22
)
T1, T2, R1, R2, Q11, Q12, Q21, Q22 = get_bounding_box(T, R_lambda_B, pipe_diameter_factors)
latex_aD, aD = interpolate_bilinear_with_fallback(
    T=T, R=R_lambda_B,
    T1=T1, T2=T2,
    R1=R1, R2=R2,
    Q11=Q11, Q12=Q12,
    Q21=Q21, Q22=Q22
)

# Output results
KH = calculate_specific_heat_output(B0, a_b, aT, aD, au, m_t, m_u, m_D) * room_size
delta_theta = room_heatload / KH
theta_RL = berechne_ruecklauf(delta_theta, theta_VL, theta_room)

# Recalculate for log. mean temperature
delta_theta_arm = (theta_VL - theta_room) / (theta_RL - theta_room)
delta_theta_log = (theta_VL - theta_RL) / np.log((theta_VL - theta_room) / (theta_RL - theta_room))
heat_output = KH * delta_theta_log
mass_flow_rate = berechne_massestrom(heat_output, theta_VL, theta_RL, c_water)
density = water_density(theta_VL - theta_RL)
flow_l_min = (mass_flow_rate / density) / 60

# --- Output Metrics ---
st.divider()
st.header("📊 Ergebnisse")

col1, col2, col3 = st.columns(3)
col1.metric("Übertemperatur Δθ", f"{delta_theta:.2f} K")
col2.metric("Rücklauf-Temp.", f"{theta_RL:.2f} °C")
col3.metric("Spreizung", f"{theta_VL - theta_RL:.2f} K")

col4, col5, col6 = st.columns(3)
col4.metric("Heizleistung", f"{heat_output:.2f} W")
col5.metric("Massestrom", f"{mass_flow_rate:.2f} kg/h")
col6.metric("Volumenstrom", f"{flow_l_min:.2f} l/min")

st.caption(
    f"Arithmetische Übertemperatur: {(theta_VL + theta_RL) / 2 - theta_room:.2f} K | "
    f"c-Faktor: {delta_theta_arm:.2f} ➝ "
    f"{'arithmetisch' if delta_theta_arm < 0.7 else 'logarithmisch'} verwenden"
)

# --- Detailed Tabs ---
st.divider()
tabs = st.tabs(["Fußbodenfaktor a_b", "m-Faktoren", "at", "au", "ad"])

with tabs[0]:
    st.subheader("Fußbodenfaktor a_b")
    latex_ab = latex_ab.replace("R_{lambda_{B}}", "R_{\\lambda,B}")
    latex_ab = latex_ab.replace(r"s_{u}", r"s_{u,0}")
    latex_ab = latex_ab.replace(r"lambdaE", r"\lambda_{E}")
    latex_ab = latex_ab.replace(r"lambdau0", r"\lambda_{u,0}")
    st.latex(latex_ab)

with tabs[1]:
    st.subheader("m-Faktoren")
    st.latex(latex_mt)
    st.latex(latex_mu)
    st.latex(latex_md)
modal = Modal(
    "Demo Modal", 
    key="demo-modal",
    # Optional
    padding=20,    # default value
    max_width=744  # default value
)
with tabs[2]:
    st.subheader("Überdeckungsfaktor aT (linear)")
    st.write("Interpoliert nach Rohrabstand und Rλ,B")
    st.latex(latex_aT)
    st.image("images/at.png", caption="DIN EN 1264 Tabelle A.1: Rohrabstand-Faktor aT")

with tabs[3]:
    st.subheader("Überdeckungsfaktor au (bilinear)")
    st.latex(latex_au)
    st.image("images/au.png", caption="DIN EN 1264 Tabelle A.2: Überdeckungsfaktor au")

with tabs[4]:
    st.subheader("Rohraußendurchmesser-Faktor aD (bilinear)")
    st.latex(latex_aD)
    st.image("images/ad.png", caption="DIN EN 1264 Tabelle A.3: Rohraußendurchmesser-Faktor aD")
