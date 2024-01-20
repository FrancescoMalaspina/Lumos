# third party imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.constants import c
import matplotlib.pyplot as plt

# local imports
from src.sympy.photonic_molecule import SymPy_PhotonicMolecule
from src.sympy.ring_resonator import SymPy_RingResonator

PM = SymPy_PhotonicMolecule()
ring = SymPy_RingResonator()

st.set_page_config(layout='wide')
st.title('Photonic Molecule')

pin = st.sidebar.selectbox('Pin', [0, 1, 2, 3, 4, 5, 6, 7], index=1)
# onluy integer values are allowed for m, n, k, min_value=1, step=1
m_1 = st.sidebar.number_input('m', value=1, min_value=1, step=1)
m_2 = st.sidebar.number_input('n', value=1, min_value=1, step=1)
p = st.sidebar.number_input('p', value=1, min_value=1, step=1)
cross_coupling_1 = st.sidebar.number_input('cross_coupling_1', value=0.5, min_value=0., max_value=1., step=0.01)
cross_coupling_2 = st.sidebar.number_input('cross_coupling_2', value=0.48, min_value=0., max_value=1., step=0.01)
unitary_loss_coefficient = st.sidebar.number_input('unitary_loss_coefficient', value=0.999, min_value=0., max_value=1.,  format='%.3f')

pm_params = {
    'm_1': m_1,
    'm_2': m_2,
    'p': p,
    'cross_coupling_1': cross_coupling_1,
    'unitary_loss_coefficient': unitary_loss_coefficient,
    "cross_coupling_2": cross_coupling_2,
    'self_coupling_1': np.sqrt(1 - cross_coupling_1**2),
    'self_coupling_2': np.sqrt(1 - cross_coupling_2**2),
}

ring_params = {
    'l': m_1+m_2,
    'cross_coupling_1': cross_coupling_1,
    'self_coupling_1': np.sqrt(1 - cross_coupling_1**2),
    'unitary_loss_coefficient': unitary_loss_coefficient,
}

PM.numeric_parameters = pm_params
ring.numeric_parameters = ring_params

fig, ax = plt.subplots(1, 2, figsize = (20, 9))
PM.pole_zero_plot(pin=1, ax=ax[0], fig=fig)
PM.magnitude_response_plot(pin=pin, ax=ax[1], fig=fig, label='PM', omega=np.linspace(-np.pi, np.pi, 10000))
ring.magnitude_response_plot(pin=pin, ax=ax[1], fig=fig, is_reference=True, omega=np.linspace(-np.pi, np.pi, 10000))
# plt.show()

st.pyplot(fig)