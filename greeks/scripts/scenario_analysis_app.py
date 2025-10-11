import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from plotly import graph_objs as go

from utils import bsm_greeks, create_greeks_df
from visual import plot_scenario_analysis

# ======== STREAMLIT UI ========

st.title("< Black-Scholes-Merton Scenario Analysis >")

S = st.number_input("Spot (S)", 100.0)
K = st.number_input("Strike (K)", 100.0)
r = st.number_input("Rate (r)", 0.05)
q = st.number_input("Dividend (q)", 0.0)
v = st.number_input("Volatility (v)", 0.15)
T = st.number_input("Maturity (T in years)", 1.0)
call_or_put = st.selectbox("Call/Put", ["call", "put"])
vary = st.selectbox("Vary Parameter", ["S", "v", "T"])
low_b = st.number_input("Lower bound (%)", -0.5)
up_b  = st.number_input("Upper bound (%)", 1.0)
step  = st.number_input("Step (%)", 0.05)

if st.button("Generate Plot"):
    base_params = dict(S=S, K=K, r=r, q=q, v=v, T=T, call_or_put=call_or_put)
    df = create_greeks_df(vary, low_b, up_b, step, base_params)
    fig = plot_scenario_analysis(df, vary, base_params, save_path=None)
    st.plotly_chart(fig, use_container_width=True)

# streamlit run scenario_analysis_app.py
