# third party imports
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks

# local imports
from sympy.physics.control.control_plots import pole_zero_numerical_data

# type hinting
from sympy.physics.control.lti import TransferFunction

def pole_zero_plot(
        system: TransferFunction, 
        pole_color: str = 'red', 
        pole_markersize: float  = 12, 
        zero_color: str = 'blue', 
        zero_markersize: float = 10, 
        fig: go.Figure = go.Figure(), 
        grid: bool = True, 
        show: bool = False
    ):
    r"""
    Returns the Pole-Zero plot (also known as PZ Plot or PZ Map) of a system as a plotly Figure object.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ----------
    system : TransferFunction
        The system for which the Pole-Zero plot is to be generated.
    pole_color : str, optional
        The color of the pole markers. The default is 'red'.
    pole_markersize : float, optional
        The size of the pole markers. The default is 12.
    zero_color : str, optional
        The color of the zero markers. The default is 'blue'.
    zero_markersize : float, optional
        The size of the zero markers. The default is 10.
    fig : go.Figure, optional
        A plotly Figure object. If None, a new Figure object is created. The default is go.Figure().
    grid : bool, optional
        Whether to show the grid. The default is True.
    show : bool, optional
        Whether to show the figure. The default is False.
    
    Returns
    -------
    fig : go.Figure
        The Pole-Zero plot of the system.
    """
    zeros, poles = pole_zero_numerical_data(system)

    zero_real = np.real(zeros)
    zero_imag = np.imag(zeros)

    pole_real = np.real(poles)
    pole_imag = np.imag(poles)

    # add poles trace
    fig.add_trace(go.Scatter(x=pole_real, y=pole_imag, mode='markers', marker_symbol='x', marker=dict(size=pole_markersize, color=pole_color), name='Poles'))

    # add zeros trace
    fig.add_trace(go.Scatter(x=zero_real, y=zero_imag, mode='markers', marker=dict(size=zero_markersize, color=zero_color), name='Zeros'))

    # set layout
    fig.update_layout(
        title='Pole-Zero Plot',
        xaxis_title="Real Axis",
        yaxis_title="Imaginary Axis",
    )

    if grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Gray')

    if show:
        fig.show()
        return

    return fig

def peak_height(magnitude_response):
    peaks, _ = find_peaks(magnitude_response)
    peak_heights = magnitude_response[peaks]
    return peak_heights[0]

def compute_fwhm(peak_index, peak_heigth, x_data, y_data):
    fwhm = []
    left_idx = peak_index
    while left_idx > 0 and y_data[left_idx] > peak_heigth / 2:
        left_idx -= 1
    right_idx = peak_index
    while right_idx < len(y_data) - 1 and y_data[right_idx] > peak_heigth / 2:
        right_idx += 1
    left_x = np.interp(peak_heigth/2, [y_data[left_idx], y_data[left_idx+1]], [x_data[left_idx], x_data[left_idx+1]])
    right_x = np.interp(peak_heigth/2, [y_data[right_idx], y_data[right_idx-1]], [x_data[right_idx], x_data[right_idx-1]])
    fwhm = right_x - left_x
    return fwhm