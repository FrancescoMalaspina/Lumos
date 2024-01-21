# third party imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# local imports
from src.sympy.interferometric_coupler import SymPy_InterferometricCoupler
from src.sympy.ring_resonator import SymPy_RingResonator

IC = SymPy_InterferometricCoupler()
ring = SymPy_RingResonator()

st.set_page_config(layout='wide')
st.title('Interferometric Coupler (DCDR)')

pin = st.sidebar.selectbox('Pin', [0, 1, 2, 3, 4, 5, 6, 7], index=1)
# only integer values are allowed for m, n, k, min_value=1, step=1
m = st.sidebar.number_input('$m$', value=1, min_value=1, step=1)
n = st.sidebar.number_input('$n$', value=3, min_value=1, step=1)
p = st.sidebar.number_input('$p$', value=4, min_value=1, step=1)
cross_coupling_1 = st.sidebar.number_input(r'$\kappa_1$', value=0.5, min_value=0., max_value=1., step=0.01)
cross_coupling_2 = st.sidebar.number_input(r'$\kappa_2$', value=0.48, min_value=0., max_value=1., step=0.01)
unitary_loss_coefficient = st.sidebar.number_input(r'$\gamma$', value=0.999, min_value=0., max_value=1.,  format='%.3f')

ic_params = {
    'm': m,
    'n': n,
    'p': p,
    'cross_coupling_1': cross_coupling_1,
    'unitary_loss_coefficient': unitary_loss_coefficient,
    "cross_coupling_2": cross_coupling_2,
    'self_coupling_1': np.sqrt(1 - cross_coupling_1**2),
    'self_coupling_2': np.sqrt(1 - cross_coupling_2**2),
}

ring_params = {
    'l': m+n,
    'cross_coupling_1': cross_coupling_1,
    'self_coupling_1': np.sqrt(1 - cross_coupling_1**2),
    'unitary_loss_coefficient': unitary_loss_coefficient,
}

IC.numeric_parameters = ic_params
ring.numeric_parameters = ring_params

pole_zero_plot = go.Figure()
pole_zero_plot = IC.plotly_pole_zero_plot(pin = pin, fig = pole_zero_plot)

magnitude_response_plot = go.Figure()
magnitude_response_plot = IC.plotly_magnitude_response_plot(pin=pin, label='DCDR', omega=np.linspace(0, 2*np.pi, 10000), fig=magnitude_response_plot)
magnitude_response_plot = ring.plotly_magnitude_response_plot(pin=pin, label='Reference Ring Resonator', omega=np.linspace(0, 2*np.pi, 10000), fig=magnitude_response_plot, is_reference=True)

col1, col2 = st.columns(2)
col1.plotly_chart(pole_zero_plot)
col2.plotly_chart(magnitude_response_plot)