# Import necessary libraries
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src import wavelength_to_frequency, HeadlessSnowman, HeadlessSnowmanInternalSource, HeadlessSnowman_withPhaseDelay
from scipy.constants import c

# structure parameters
main_radius = 120 * 1e-6
auxiliary_radius = 90 * 1e-6
mach_zender_length = 120 * np.pi * 1e-6
main_cross_coupling = 1e-1
auxiliary_cross_coupling = 3e-1

# propagation parameters
central_wavelength = 1.55e-6
effective_index = 1.7
group_index = 2
GVD = 0.6e-24
loss_dB = 10 # 10 dB/m

# simulation parameters
number_of_points = 5000
omega_0 = wavelength_to_frequency(1550e-9)
omega_m = c / main_radius / group_index
n_res = 5
angular_frequencies = np.linspace(omega_0 - n_res/2*omega_m, omega_0 + n_res/2*omega_m, number_of_points)
pin = 1


# interactive plot with plotly
def interactive_plot(
        MZ_ratio, 
        auxiliary_radius_ratio, 
        new_main_cross_coupling, 
        new_trough_cross_coupling,
        new_auxiliary_cross_coupling, 
        new_phase_delay,
        dB_scale, 
        n_res_left, 
        n_res_right, 
        number_of_points,
        pin
    ): 
    angular_frequencies = np.linspace(omega_0 + n_res_left*omega_m, omega_0 + n_res_right*omega_m, number_of_points)
    
    # objects
    reference_HS = HeadlessSnowman(
        main_radius = main_radius,
        auxiliary_radius = auxiliary_radius,
        mach_zender_length = mach_zender_length,
        input_cross_coupling_coefficient=main_cross_coupling,
        through_cross_coupling_coefficient=main_cross_coupling,
        ring_cross_coupling_coefficient= 0, 
        effective_refractive_index=effective_index,
        group_refractive_index=group_index,
        GVD=GVD,
        loss_dB=loss_dB,
        central_wavelength=central_wavelength,
        angular_frequencies=angular_frequencies,
    )
    new_HS = HeadlessSnowman_withPhaseDelay(
        main_radius = main_radius,
        auxiliary_radius = auxiliary_radius_ratio * main_radius,
        mach_zender_length = MZ_ratio * main_radius * np.pi,
        input_cross_coupling_coefficient = new_main_cross_coupling,
        through_cross_coupling_coefficient = new_trough_cross_coupling,
        ring_cross_coupling_coefficient = new_auxiliary_cross_coupling,
        effective_refractive_index = effective_index,
        group_refractive_index = group_index,
        GVD = GVD,
        loss_dB = loss_dB,
        central_wavelength = central_wavelength,
        angular_frequencies = angular_frequencies,
        phase_delay = new_phase_delay
    )
    reference_internal_source_HS = HeadlessSnowmanInternalSource(
        main_radius = main_radius,
        auxiliary_radius = auxiliary_radius,
        mach_zender_length = mach_zender_length,
        input_cross_coupling_coefficient=main_cross_coupling,
        through_cross_coupling_coefficient=main_cross_coupling,
        ring_cross_coupling_coefficient= 0, 
        effective_refractive_index=effective_index,
        group_refractive_index=group_index,
        GVD=GVD,
        loss_dB=loss_dB,
        central_wavelength=central_wavelength,
        angular_frequencies=angular_frequencies,
    )
    internal_source_HS = HeadlessSnowmanInternalSource(
        main_radius = main_radius,
        auxiliary_radius = auxiliary_radius_ratio * main_radius,
        mach_zender_length = MZ_ratio * main_radius * np.pi,
        input_cross_coupling_coefficient = new_main_cross_coupling,
        through_cross_coupling_coefficient = new_trough_cross_coupling,
        ring_cross_coupling_coefficient = new_auxiliary_cross_coupling,
        effective_refractive_index = effective_index,
        group_refractive_index = group_index,
        GVD = GVD,
        loss_dB = loss_dB,
        central_wavelength = central_wavelength,
        angular_frequencies = angular_frequencies,
    )   
    if dB_scale:
        modulus_fig = go.Figure()
        reference_fields = reference_HS.fields
        new_fields = new_HS.fields
        modulus_fig.add_trace(go.Scatter(x=angular_frequencies, y=10*np.log10(np.abs(reference_fields[:, pin])**2), mode='lines', name='reference', line=dict(color="#1f77b4", dash='dot', width=2)))
        modulus_fig.add_trace(go.Scatter(x=angular_frequencies, y=10*np.log10(np.abs(new_fields[:, pin])**2), mode='lines', name=f'HS signal', line=dict(color="#ff7f0e",  width=2)))
        modulus_fig.update_layout(xaxis_title='Angular frequency [rad/s]', yaxis_title='Field enhancement [dB]', autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))
    else:
        modulus_fig = go.Figure()
        reference_fields = reference_HS.fields
        new_fields = new_HS.fields
        modulus_fig.add_trace(go.Scatter(x=angular_frequencies, y=np.abs(reference_fields[:, pin]), mode='lines', name='reference', line=dict(color="#1f77b4", dash='dot', width=2)))
        modulus_fig.add_trace(go.Scatter(x=angular_frequencies, y=np.abs(new_fields[:, pin]), mode='lines', name=f'HS signal', line=dict(color="#ff7f0e",  width=2)))
        modulus_fig.update_layout(xaxis_title='Angular frequency [rad/s]', yaxis_title='Field enhancement', autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))                

    # MZ phase difference
    ref_delta_phi_mz = (np.angle(reference_fields[:, 4]) - np.angle(reference_fields[:, 2])) - (np.angle(reference_fields[:, 5]) - np.angle(reference_fields[:, 3]))
    new_delta_phi_mz = (np.angle(new_fields[:, 4]) - np.angle(new_fields[:, 2])) - (np.angle(new_fields[:, 5]) - np.angle(new_fields[:, 3]))
    mz_phase_fig = go.Figure()
    mz_phase_fig.add_trace(go.Scatter(x=angular_frequencies, y=np.cos(ref_delta_phi_mz), mode='lines', name='reference', line=dict(color="#1f77b4", dash='dot', width=2)))
    mz_phase_fig.add_trace(go.Scatter(x=angular_frequencies, y=np.cos(new_delta_phi_mz), mode='lines', name=f'HS signal', line=dict(color="#ff7f0e",  width=2)))
    mz_phase_fig.update_layout(xaxis_title='Angular frequency [rad/s]', yaxis_title='cos( Phase [rad] )', autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))
    mz_phase_fig.update_xaxes(range=[angular_frequencies[0], angular_frequencies[-1]])

    # field intensity
    if dB_scale:
        intensity_fig = go.Figure()
        intensity_fig.add_trace(go.Scatter(x=angular_frequencies, y=10*np.log10(np.abs(reference_fields[:, pin])**2), mode='lines', name='reference', line=dict(color="#1f77b4", dash='dot', width=2)))
        intensity_fig.add_trace(go.Scatter(x=angular_frequencies, y=10*np.log10(np.abs(new_fields[:, pin])**2), mode='lines', name=f'HS signal', line=dict(color="#ff7f0e",  width=2)))
        intensity_fig.update_layout(xaxis_title='Angular frequency [rad/s]', yaxis_title='Intensity enhancement [dB]', autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))
    else:
        intensity_fig = go.Figure()
        intensity_fig.add_trace(go.Scatter(x=angular_frequencies, y=np.abs(reference_fields[:, pin])**2, mode='lines', name='reference', line=dict(color="#1f77b4", dash='dot', width=2)))
        intensity_fig.add_trace(go.Scatter(x=angular_frequencies, y=np.abs(new_fields[:, pin])**2, mode='lines', name=f'HS signal', line=dict(color="#ff7f0e",  width=2)))
        intensity_fig.update_layout(xaxis_title='Angular frequency [rad/s]', yaxis_title='Intensity enhancement', autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))

    # field modulus with the internal source
    if dB_scale:
        internal_source_fig = go.Figure()
        internal_source_fig.add_trace(go.Scatter(x=angular_frequencies, y=10*np.log10(np.abs(reference_internal_source_HS.fields[:, pin])**2), mode='lines', name='reference', line=dict(color="#1f77b4", dash='dot', width=2)))
        internal_source_fig.add_trace(go.Scatter(x=angular_frequencies, y=10*np.log10(np.abs(internal_source_HS.fields[:, pin])**2), mode='lines', name=f'HS signal', line=dict(color="#ff7f0e",  width=2)))
        internal_source_fig.update_layout(xaxis_title='Angular frequency [rad/s]', yaxis_title='Field enhancement [dB]', autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))
    else:
        internal_source_fig = go.Figure()
        internal_source_fig.add_trace(go.Scatter(x=angular_frequencies, y=np.abs(reference_internal_source_HS.fields[:, pin]), mode='lines', name='reference', line=dict(color="#1f77b4", dash='dot', width=2)))
        internal_source_fig.add_trace(go.Scatter(x=angular_frequencies, y=np.abs(internal_source_HS.fields[:, pin]), mode='lines', name=f'HS signal', line=dict(color="#ff7f0e",  width=2)))
        internal_source_fig.update_layout(xaxis_title='Angular frequency [rad/s]', yaxis_title='Field enhancement', autosize=False, width=800, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=4))

    return modulus_fig, intensity_fig, mz_phase_fig, internal_source_fig


# Streamlit app
def app():
    # Set page title
    # st.set_page_config(page_title='Numerical Simulation')
    st.set_page_config(layout='wide')
    st.title('Resonant Interferometric Coupler (RIC)')

    # input widgets
    dB_scale                    = st.sidebar.toggle('dB scale', value=False)
    st.sidebar.latex(r"\text{Main radius } (R_1): " + f"{main_radius:.1e} \ m")
    MZ_ratio                    = st.sidebar.number_input('MZ length ratio (MZ / pi * R_1)',    min_value=0.01,                 value=1.,           step=1e-6,  format='%.6f')
    auxiliary_radius_ratio      = st.sidebar.number_input('Auxiliary radius ratio (R_2 / R_1)', min_value=0.01,                 value=0.1,          step=1e-6,  format='%.6f')
    input_cross_coupling        = st.sidebar.number_input('Input cross coupling (κ_0)',         min_value=0.,   max_value=1.,   value=0.2,          step=0.05,  format='%.2f')
    through_cross_coupling      = st.sidebar.number_input('Through cross coupling (κ_4)',       min_value=0.,   max_value=1.,   value=0.1,          step=0.05,  format='%.2f')
    auxiliary_cross_coupling    = st.sidebar.number_input('Auxiliary cross coupling (κ_8)',     min_value=0.,   max_value=1.,   value=0.5,          step=0.05,  format='%.2f')
    phase_delay_over_pi         = st.sidebar.number_input('Phase delay (rad/$\pi$)',            min_value=0.,   max_value=2.,   value=0.,           step=0.01,  format='%.2f')
    n_res_left, n_res_right     = st.sidebar.slider('X axis range [FSR from ω_0]',              min_value=-20., max_value=20.,  value=(-0., +6.),   step=0.1,   format='%f')
    pin                         = st.sidebar.selectbox('Pin', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], index=1)
    number_of_points            = st.sidebar.number_input('Number of points', min_value=1000, max_value=50000, value=5000, step=1000, format='%d')

    # Build the interactive plots
    modulus_fig, intensity_fig, phase_fig, internal_source_fig = interactive_plot(
        MZ_ratio, 
        auxiliary_radius_ratio, 
        input_cross_coupling, 
        through_cross_coupling,
        auxiliary_cross_coupling, 
        phase_delay_over_pi * np.pi,
        dB_scale, 
        n_res_left, 
        n_res_right, 
        number_of_points,
        pin
    )


    # Display the plots
    with st.expander(f'**Field enhancement @ pin {pin}**', expanded=True):
        st.plotly_chart(modulus_fig, use_container_width=True)
    with st.expander(f'**Intensity enhancement @ pin {pin}**', expanded=False):
        st.plotly_chart(intensity_fig, use_container_width=True)
    with st.expander(f'**Cosine of the phase delay in the Mach Zender**', expanded=True):
        st.plotly_chart(phase_fig, use_container_width=True)
    with st.expander(f'**Internal Source: Field enhancement @ pin {pin}**', expanded=True):
        st.plotly_chart(internal_source_fig, use_container_width=True)
    diagram = 'img/tikz.png'
    with st.expander('**Structure diagram**', expanded=True):
        cols = st.columns([1, 1, 1])
        cols[1].image(diagram, width=500)
        diagram = 'img/tikz.png'
    internal_source_diagram = 'img/internal_source.png'
    with st.expander('**Structure diagram with the internal source**', expanded=False):
        cols = st.columns([1, 1, 1])
        cols[1].image(internal_source_diagram, width=500)


# Run the app
if __name__ == '__main__':
    app()
