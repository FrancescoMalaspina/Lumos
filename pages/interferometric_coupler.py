# third party imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.constants import c
import matplotlib.pyplot as plt

# local imports
from src.sympy.interferometric_coupler import SymPy_InterferometricCoupler
from src.sympy.ring_resonator import SymPy_RingResonator

IC = SymPy_InterferometricCoupler()
ring = SymPy_RingResonator()

pin = st.sidebar.selectbox('Pin', [0, 1, 2, 3, 4, 5, 6, 7], index=1)
# onluy integer values are allowed for m, n, k, min_value=1, step=1
m = st.sidebar.number_input('m', value=1, min_value=1, step=1)
n = st.sidebar.number_input('n', value=3, min_value=1, step=1)
k = st.sidebar.number_input('k', value=4, min_value=1, step=1)
cross_coupling_1 = st.sidebar.number_input('cross_coupling_1', value=0.5, min_value=0., max_value=1., step=0.01)
cross_coupling_2 = st.sidebar.number_input('cross_coupling_2', value=0.48, min_value=0., max_value=1., step=0.01)
unitary_loss_coefficient = st.sidebar.number_input('unitary_loss_coefficient', value=0.999, min_value=0., max_value=1.,  format='%.3f')


ic_params = {
    'm': m,
    'n': n,
    'k': k,
    'cross_coupling_1': cross_coupling_1,
    'cross_coupling_2': cross_coupling_2,
    'unitary_loss_coefficient': unitary_loss_coefficient,
    'pin': pin,
}

ring_params = {
    'l': m+n,
    'cross_coupling_1': cross_coupling_1,
    'unitary_loss_coefficient': unitary_loss_coefficient,
    'pin': pin%4,
}

fig1, ax1 = plt.subplots(figsize = (6, 6))
IC.pole_zero_plot(**ic_params, ax=ax1, fig=fig1)
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize = (6, 6))
IC.magnitude_response_plot(**ic_params, ax=ax2, fig=fig2)
ring.magnitude_response_plot(**ring_params, ax=ax2, fig=fig2)
st.pyplot(fig2)
