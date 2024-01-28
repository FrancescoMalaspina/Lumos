# third party imports
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# local imports
from src.sympy.resonant_interferometric_coupler import SymPy_ResonantInterferometricCoupler
from src.sympy.ring_resonator import SymPy_RingResonator

RIC = SymPy_ResonantInterferometricCoupler()
ring = SymPy_RingResonator()

st.set_page_config(layout='wide')
st.title('Resonant Interferometric Coupler (RIC)')

pin = st.sidebar.selectbox('Pin', [i for i in range(12)], index=1)
# onluy integer values are allowed for m, n, k, min_value=1, step=1
m_1 = st.sidebar.number_input('$m_1$', value=2, min_value=1, step=1)
m_2 = st.sidebar.number_input('$m_2$', value=2, min_value=1, step=1)
n_1 = st.sidebar.number_input('$n_1$', value=1, min_value=1, step=1)
n_2 = st.sidebar.number_input('$n_2$', value=1, min_value=1, step=1)
p = st.sidebar.number_input('$p$', value=3, min_value=1, step=1)
cross_coupling_1 = st.sidebar.number_input(r'$\kappa_1$', value=0.2, min_value=0., max_value=1., step=0.01)
cross_coupling_2 = st.sidebar.number_input(r'$\kappa_2$', value=0.2, min_value=0., max_value=1., step=0.01)
cross_coupling_a = st.sidebar.number_input(r'$\kappa_a$', value=0.4, min_value=0., max_value=1., step=0.01)
unitary_loss_coefficient = st.sidebar.number_input(r'$\gamma$', value=0.999, min_value=0., max_value=1.,  format='%.3f')

ric_params = {
    'm_1': m_1,
    'm_2': m_2,
    'n_1': n_1,
    'n_2': n_2,
    'p': p,
    'cross_coupling_1': cross_coupling_1,
    'cross_coupling_2': cross_coupling_2,
    'cross_coupling_a': cross_coupling_a,
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

pole_zero_plot = go.Figure()
pole_zero_plot = RIC.plotly_pole_zero_plot(pin = pin, fig = pole_zero_plot)

magnitude_response_plot = go.Figure()
magnitude_response_plot = RIC.plotly_magnitude_response_plot(
    pin=pin, label='Resonant Interferometric Coupler', 
    omega=np.linspace(-np.pi, np.pi, 10000), 
    fig=magnitude_response_plot
    )
if pin <4:
    magnitude_response_plot = ring.plotly_magnitude_response_plot(
        pin=pin, label='Reference Ring Resonator', 
        omega=np.linspace(-np.pi, np.pi, 10000), 
        fig=magnitude_response_plot, 
        is_reference=True)

magnitude_response_plot.update_layout(
legend=dict(
    orientation="h",
    # yanchor="bottom",
    # y=1.02,
    # xanchor="right",
    # x=1
))

pole_zero_plot.update_layout(
    legend = dict(
        orientation='h'
    )
)

col1, col2 = st.columns(2)
col1.plotly_chart(pole_zero_plot, use_container_width=True)
col2.plotly_chart(magnitude_response_plot, use_container_width=True)