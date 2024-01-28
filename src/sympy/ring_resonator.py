# third party imports
from sympy import symbols, Eq, I
import numpy as np
import plotly.graph_objects as go

# local imports
from src.sympy.base import SymPy_PhotonicCircuit

# type hinting
from typing import Any

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
    
    def plotly_magnitude_response_plot(
            self, 
            pin: int, 
            label: str = None, 
            is_reference: bool = False,
            omega: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(0, 2*np.pi, 10000),
            fig: go.Figure = go.Figure(),
    ) -> go.Figure:
        
        # check if numeric parameters are set
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling magnitude_response_plot")
        
        # plot
        magnitude_response_lambda = self.numeric_lambda_solution(pin)
        magnitude_response = np.abs(magnitude_response_lambda(np.exp(1j * omega)))

        if is_reference:
            fig.add_trace(go.Scatter(x=omega, y=magnitude_response, mode='lines', name='Reference Ring Resonator', line=dict(dash='dash')))
        else:
            if label is None:
                fig.add_trace(go.Scatter(x=omega, y=magnitude_response, mode='lines'))
            else:
                fig.add_trace(go.Scatter(x=omega, y=magnitude_response, mode='lines', name=label))

        # set layout
        fig.update_layout(
            title='Magnitude response',
            xaxis_title="Normalized ω [rad/s]",
            yaxis_title=f"H_{pin}(ω)",
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
        )

        return fig