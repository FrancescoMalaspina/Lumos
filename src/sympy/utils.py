from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sympy.physics.control.control_plots import pole_zero_numerical_data
import matplotlib.pyplot as plt
import numpy as np

def pole_zero_plot(system, pole_color='red', pole_markersize=10, zero_color='blue', zero_markersize=7, grid=True, show=False, fig: Figure = None, ax: Axes = None):
    r"""
    Returns the Pole-Zero plot (also known as PZ Plot or PZ Map) of a system.

    A Pole-Zero plot is a graphical representation of a system's poles and
    zeros. It is plotted on a complex plane, with circular markers representing
    the system's zeros and 'x' shaped markers representing the system's poles.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type systems
        The system for which the pole-zero plot is to be computed.
    pole_color : str, tuple, optional
        The color of the pole points on the plot. Default color
        is blue. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    pole_markersize : Number, optional
        The size of the markers used to mark the poles in the plot.
        Default pole markersize is 10.
    zero_color : str, tuple, optional
        The color of the zero points on the plot. Default color
        is orange. The color can be provided as a matplotlib color string,
        or a 3-tuple of floats each in the 0-1 range.
    zero_markersize : Number, optional
        The size of the markers used to mark the zeros in the plot.
        Default zero markersize is 7.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    zeros, poles = pole_zero_numerical_data(system)

    zero_real = np.real(zeros)
    zero_imag = np.imag(zeros)

    pole_real = np.real(poles)
    pole_imag = np.imag(poles)

    ax.plot(pole_real, pole_imag, 'x', mfc='none', markersize=pole_markersize, color=pole_color)
    ax.plot(zero_real, zero_imag, 'o', markersize=zero_markersize, color=zero_color)
    ax.set_xlabel('Real Axis')
    ax.set_ylabel('Imaginary Axis')

    if grid:
        ax.grid(True)
    if show:
        plt.show()
        return

    return fig