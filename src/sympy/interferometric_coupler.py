from src.sympy.base import SymPy_PhotonicCircuit

from sympy import symbols, Eq, I

class SymPy_InterferometricCoupler(SymPy_PhotonicCircuit):
    num_pins = 8

    def __init__(self, numeric_parameters=None):
        super().__init__(numeric_parameters=numeric_parameters)
    
    @property
    def parameter_symbols(self):
        return {
            "m": symbols("m"),
            "n": symbols("n"),
            "p": symbols("p"),
            "cross_coupling_1": symbols(r"\kappa_1"),
            "cross_coupling_2": symbols(r"\kappa_2"),
            "self_coupling_1": symbols(r"\sigma_1"),
            "self_coupling_2": symbols(r"\sigma_2"),
            "unitary_loss_coefficient": symbols(r"\gamma"),
        }
    
    @property
    def equations(self):
        a0, a1, a2, a3, a4, a5, a6, a7 = self.pins
        z = self.z
        return [
            Eq(a0, 1),
            Eq(a2, self.parameter_symbols["self_coupling_1"] * a0 + I * self.parameter_symbols["cross_coupling_1"] * a1),
            Eq(a3, I * self.parameter_symbols["cross_coupling_1"] * a0 + self.parameter_symbols["self_coupling_1"] * a1),
            Eq(a6, self.parameter_symbols["self_coupling_2"] * a4 + I * self.parameter_symbols["cross_coupling_2"] * a5),
            Eq(a7, I * self.parameter_symbols["cross_coupling_2"] * a4 + self.parameter_symbols["self_coupling_2"] * a5),
            Eq(a5, a3 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["n"]),
            Eq(a1, a7 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["m"]),
            Eq(a4, a2 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["p"]),
        ]