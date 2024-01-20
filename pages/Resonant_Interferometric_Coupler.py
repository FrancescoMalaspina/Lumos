# third party imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.constants import c
import matplotlib.pyplot as plt

# local imports
from src.sympy.resonant_interferometric_coupler import SymPy_ResonantInterferometricCoupler
from src.sympy.ring_resonator import SymPy_RingResonator

RIC = SymPy_ResonantInterferometricCoupler()
ring = SymPy_RingResonator()

st.set_page_config(layout='wide')
st.title('Resonant Interferometric Coupler (RIC)')

pin = st.sidebar.selectbox('Pin', [0, 1, 2, 3, 4, 5, 6, 7], index=1)
# onluy integer values are allowed for m, n, k, min_value=1, step=1
m_1 = st.sidebar.number_input('$m_1$', value=2, min_value=1, step=1)
m_2 = st.sidebar.number_input('$m_2$', value=2, min_value=1, step=1)
n_1 = st.sidebar.number_input('n_1', value=1, min_value=1, step=1)
n_2 = st.sidebar.number_input('n_2', value=1, min_value=1, step=1)
p = st.sidebar.number_input('p', value=3, min_value=1, step=1)
cross_coupling_1 = st.sidebar.number_input('cross_coupling_1', value=0.2, min_value=0., max_value=1., step=0.01)
cross_coupling_2 = st.sidebar.number_input('cross_coupling_2', value=0.2, min_value=0., max_value=1., step=0.01)
cross_coupling_a = st.sidebar.number_input('cross_coupling_a', value=0.4, min_value=0., max_value=1., step=0.01)
unitary_loss_coefficient = st.sidebar.number_input('unitary_loss_coefficient', value=0.999, min_value=0., max_value=1.,  format='%.3f')

ric_params = {
    'm_1': m_1,
    'm_2': m_2,
    'n_1': n_1,
    'n_2': n_2,
    'p': p,
    'cross_coupling_1': cross_coupling_1,
    "cross_coupling_2": cross_coupling_2,
    "cross_coupling_a": cross_coupling_a,
    'self_coupling_1': np.sqrt(1 - cross_coupling_1**2),
    'self_coupling_2': np.sqrt(1 - cross_coupling_2**2),
    'self_coupling_a': np.sqrt(1 - cross_coupling_a**2),
    'unitary_loss_coefficient': unitary_loss_coefficient,
}

ring_params = {
    'l': m_1+m_2,
    'cross_coupling_1': cross_coupling_1,
    'self_coupling_1': np.sqrt(1 - cross_coupling_1**2),
    'unitary_loss_coefficient': unitary_loss_coefficient,
}

RIC.numeric_parameters = ric_params
ring.numeric_parameters = ring_params

fig, ax = plt.subplots(1, 2, figsize = (20, 9))
RIC.pole_zero_plot(pin=1, ax=ax[0], fig=fig)
RIC.magnitude_response_plot(pin=pin, ax=ax[1], fig=fig, label='RIC', omega=np.linspace(-np.pi, np.pi, 10000))
ring.magnitude_response_plot(pin=pin, ax=ax[1], fig=fig, is_reference=True, omega=np.linspace(-np.pi, np.pi, 10000))
# plt.show()

st.pyplot(fig)