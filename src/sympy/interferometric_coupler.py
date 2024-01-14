from sympy import symbols, Eq, I, linsolve, together, lambdify
from sympy.physics.control.lti import TransferFunction
from sympy.physics.control.control_plots import pole_zero_plot
import numpy as np
import matplotlib.pyplot as plt

class SymPy_InterferometricCoupler:

    def __init__(self):
        self.m = symbols("m")
        self.n = symbols("n")
        self.k = symbols("k")
        self.cross_coupling_1 = symbols(r"\kappa_1")
        self.cross_coupling_2 = symbols(r"\kappa_2")
        self.self_coupling_1 = symbols(r"\sigma_1")
        self.self_coupling_2 = symbols(r"\sigma_2")
        self.unitary_loss_coefficient = symbols(r"\gamma")
        self.z = symbols("z")
        self.pins = [symbols("A_{}".format(i)) for i in range(8)]

    @property
    def equations(self):
        a0, a1, a2, a3, a4, a5, a6, a7 = self.pins
        z = self.z
        return [
            Eq(a0, 1),
            Eq(a2, self.self_coupling_1 * a0 + I * self.cross_coupling_1 * a1),
            Eq(a3, I * self.cross_coupling_1 * a0 + self.self_coupling_1 * a1),
            Eq(a6, self.self_coupling_2 * a4 + I * self.cross_coupling_2 * a5),
            Eq(a7, I * self.cross_coupling_2 * a4 + self.self_coupling_2 * a5),
            Eq(a5, a3 * (self.unitary_loss_coefficient * z ** (-1)) ** self.n), 
            Eq(a1, a7 * (self.unitary_loss_coefficient * z ** (-1)) ** self.m),     
            Eq(a4, a2 * (self.unitary_loss_coefficient * z ** (-1)) ** self.k),  

        ]

    @property
    def solutions(self):
        solution_set = together(linsolve(self.equations, self.pins))
        return list(solution_set)[0]

    def solution(self, pin):
        return self.solutions[pin]

    def numeric_solution(self, pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient):
        return self.solution(pin).subs({
            self.m: m,
            self.n: n,
            self.k: k,
            self.cross_coupling_1: cross_coupling_1,
            self.cross_coupling_2: cross_coupling_2,
            self.self_coupling_1: np.sqrt(1 - cross_coupling_1**2),
            self.self_coupling_2: np.sqrt(1 - cross_coupling_2**2),
            self.unitary_loss_coefficient: unitary_loss_coefficient,
        })

    def numeric_solution_lambdified(self, pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient):
        return lambdify(self.z, self.numeric_solution(pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient))

    def transfer_function(self, pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient):
        return TransferFunction.from_rational_expression(self.numeric_solution(pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient), self.z)

    def pole_zero_plot(self, pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        pole_zero_plot(self.transfer_function(pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient), ax=ax, show=False)
        omega = np.linspace(0, 2*np.pi, 10000)
        # black
        ax.plot(np.cos(omega), np.sin(omega), color="black")
        ax.set_title(f'Pole-zero plot')
        return fig

    def magnitude_response_plot(self, pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        magnitude_response_lambda = self.numeric_solution_lambdified(pin, m, n, k, cross_coupling_1, cross_coupling_2, unitary_loss_coefficient)
        omega = np.linspace(0, 2*np.pi, 10000)
        magnitude_response = np.abs(magnitude_response_lambda(np.exp(1j * omega)))
        ax.plot(omega, magnitude_response)
        ax.set_title(f'Magnitude response')
        ax.set_xlabel(r"Normalized $\omega$ [rad/s]")
        ax.set_ylabel(f"$H_{pin}(\omega)$")
        ax.grid()
        return fig