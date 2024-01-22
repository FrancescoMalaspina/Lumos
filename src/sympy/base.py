# third party imports
import numpy as np
import plotly.graph_objects as go
import os
import pickle
from abc import ABC, abstractmethod

from numpy import ndarray, dtype, floating
from numpy._typing import _64Bit

# local imports
from sympy import symbols, Eq, I, linsolve, together, lambdify
from src.sympy.utils import pole_zero_plot
from src.config import SYMPY_DATA_PATH

# type hinting
from sympy.core.expr import Expr
from sympy.physics.control.lti import TransferFunction
from typing import Any, Tuple


class SymPy_PhotonicCircuit(ABC):
    num_pins = 1

    def __init__(self, numeric_parameters=None):
        if numeric_parameters is None:
            numeric_parameters = {}
        self.z = symbols("z")
        self.pins = [symbols("A_{}".format(i)) for i in range(self.num_pins)]
        self._numeric_parameters = numeric_parameters

    @property
    @abstractmethod
    def parameter_symbols(self) -> dict:
        pass

    @property
    def numeric_parameters(self) -> dict:
        return self._numeric_parameters

    @numeric_parameters.setter
    def numeric_parameters(self, value: dict):
        if not set(value.keys()) == set(self.parameter_symbols.keys()):
            raise ValueError("Keys of numeric_parameters must match keys of parameter_symbols")
        self._numeric_parameters = value

    @property
    @abstractmethod
    def equations(self):
        """
        Returns a list of sympy.Eq objects of len self.num_pins
        """
        pass

    @property
    def solutions(self) -> list[Expr]:
        """
        This property represents the solutions of the system of equations defined in the `equations` method. 
        The solutions are stored in a pickle file named after the class and located in the `SYMPY_DATA_PATH` directory.
        
        If the pickle file already exists, the solutions are loaded from the file. 
        If the file does not exist, the solutions are computed using the `linsolve` function from sympy, 
        and then saved to the pickle file for future use.
        
        Returns:
            list[Expr]: A list of sympy expressions representing the solutions of the system of equations.
        """
        directory = SYMPY_DATA_PATH
        filename = os.path.join(directory, f'{self.__class__.__name__}.pkl')
        solutions = list()
        if not os.path.exists(directory):
            os.makedirs(directory)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                solutions = pickle.load(f)
                # print(f"Loading {filename}")
        else:
            solution_set = together(linsolve(self.equations, self.pins))
            solutions = list(solution_set)[0]
            with open(filename, 'wb') as f:
                pickle.dump(solutions, f)
            # print(f"Saving {filename}")
        return solutions

    def solution(
            self,
            pin
    ) -> Expr:
        """
        Returns the solution for a given pin.
        
        Args:
            pin (int): The pin for which the solution is returned.
            
        Returns:
            Expr: The solution for the given pin.
        """
        return self.solutions[pin]

    def numeric_solution(
            self,
            pin
    ) -> Expr:

        # check if numeric parameters are set
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling numeric_solution")
        substitution_table = {self.parameter_symbols[k]: v for k, v in self.numeric_parameters.items()}
        return self.solution(pin).subs(substitution_table)

    def numeric_lambda_solution(
            self,
            pin
    ) -> Any:

        # check if numeric parameters are set
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling numeric_solution_lambdified")
        return lambdify(self.z, self.numeric_solution(pin))

    def transfer_function(
            self,
            pin
    ) -> TransferFunction:

        # check if numeric parameters are set
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling transfer_function")
        return TransferFunction.from_rational_expression(self.numeric_solution(pin), self.z)

    def plotly_pole_zero_plot(
            self,
            pin: int,
            fig: go.Figure = go.Figure(),
    ) -> go.Figure:

        # check if numeric parameters are set
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling plotly_pole_zero_plot")

        # plot
        omega = np.linspace(0, 2 * np.pi, 5000)
        pole_zero_plot(self.transfer_function(pin), fig=fig, show=False)
        fig.add_trace(
            go.Scatter(x=np.cos(omega), y=np.sin(omega), mode='lines', name='unit circle', line=dict(color='black')))

        # set layout
        fig.update_layout(
            title='Pole-zero plot',
            xaxis_title="Real Axis",
            yaxis_title="Imaginary Axis",
            autosize=False,
            width=900,
            height=900,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
        )

        return fig

    def plotly_magnitude_response_plot(
            self,
            pin: int,
            label: str = None,
            omega: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(0, 2 * np.pi, 10000),
            fig: go.Figure = go.Figure(),
    ) -> go.Figure:

        # check if numeric parameters are set
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling magnitude_response_plot")

        # plot
        magnitude_response_lambda = self.numeric_lambda_solution(pin)
        magnitude_response = np.abs(magnitude_response_lambda(np.exp(1j * omega)))

        # add trace
        fig.add_trace(go.Scatter(x=omega, y=magnitude_response, mode='lines', name=label))

        # set layout
        fig.update_layout(
            title='Magnitude response',
            xaxis_title="Normalized ω [rad/s]",
            yaxis_title=f"H_{pin}(ω)",
            autosize=False,
            width=1200,
            height=900,
            margin=dict(l=50, r=50, b=100, t=100, pad=4),
        )

        return fig

    def magnitude_response_data(
            self,
            pin: int,
            omega: np.ndarray[Any, np.dtype[np.float64]] = np.linspace(0, 2 * np.pi, 10000)
    ) -> tuple[ndarray[Any, dtype[floating[_64Bit]]], Any]:
        """
        Returns the magnitude response data for a given pin.
        
        Args:
            pin (int): The pin for which the magnitude response data is returned.
            omega (np.ndarray[Any, np.dtype[np.float64]]): The angular frequency vector.
            
        Returns:
            np.ndarray[Any, np.dtype[np.float64]]: The angular frequency vector.
            np.ndarray[Any, np.dtype[np.float64]]: The magnitude response vector.
            """

        # check if numeric parameters are set
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling magnitude_response_data")

        magnitude_response_lambda = self.numeric_lambda_solution(pin)
        magnitude_response = np.abs(magnitude_response_lambda(np.exp(1j * omega)))
        return omega, magnitude_response

    def __str__(self):
        return f"{self.__class__.__name__} object"

    def __repr__(self):
        return f"{self.__class__.__name__} object"
