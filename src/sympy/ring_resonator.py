from src.sympy.base import SymPy_PhotonicCircuit

from sympy import symbols, Eq, I, linsolve, together, lambdify
from sympy.physics.control.lti import TransferFunction
from sympy.physics.control.control_plots import pole_zero_plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class SymPy_RingResonator(SymPy_PhotonicCircuit):
    num_pins = 4

    def __init__(self, numeric_parameters=None):
        super().__init__(numeric_parameters=numeric_parameters)

    @property
    def parameter_symbols(self):
        return {
            "l": symbols("l"),
            "cross_coupling_1": symbols(r"\kappa_1"),
            "self_coupling_1": symbols(r"\sigma_1"),
            "unitary_loss_coefficient": symbols(r"\gamma"),
        }
    
    @property
    def equations(self):
        a0, a1, a2, a3 = self.pins
        z = self.z
        return [
            Eq(a0, 1),
            Eq(a2, self.parameter_symbols["self_coupling_1"] * a0 + I * self.parameter_symbols["cross_coupling_1"] * a1),
            Eq(a3, I * self.parameter_symbols["cross_coupling_1"] * a0 + self.parameter_symbols["self_coupling_1"] * a1),
            Eq(a1, a3 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["l"]),
        ]
    
    def magnitude_response_plot(self, pin, fig: Figure = None, ax: Axes = None, is_reference = False, label = None) -> Figure:
        # check if numeric parameters are set
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling magnitude_response_plot")
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        magnitude_response_lambda = self.numeric_solution_lambdified(pin)
        omega = np.linspace(0, 2*np.pi, 10000)
        magnitude_response = np.abs(magnitude_response_lambda(np.exp(1j * omega)))
        if is_reference:
            ax.plot(omega, magnitude_response, label=f'ring', linestyle='dotted')
        else: 
            if label is None:
                ax.plot(omega, magnitude_response)
            else:
                ax.plot(omega, magnitude_response, label=label)
        ax.set_title(f'Magnitude response')
        ax.set_xlabel(r"Normalized $\omega$ [rad/s]")
        ax.set_ylabel(f"$H_{pin}(\omega)$")
        ax.grid("True")
        ax.legend()
        return fig