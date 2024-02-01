from src.sympy.base import SymPy_PhotonicCircuit

from sympy import symbols, Eq, I

class SymPy_ResonantInterferometricCoupler(SymPy_PhotonicCircuit):
    num_pins = 12

    def __init__(self, numeric_parameters=None):
        super().__init__(numeric_parameters=numeric_parameters)
    
    @property
    def parameter_symbols(self):
        return {
            "m_1": symbols("m_1"),
            "m_2": symbols("m_2"),
            "n_1": symbols("n_1"),
            "n_2": symbols("n_2"),
            "p": symbols("p"),
            "cross_coupling_1": symbols(r"\kappa_1"),
            "cross_coupling_2": symbols(r"\kappa_2"),
            "cross_coupling_a": symbols(r"\kappa_a"),
            "self_coupling_1": symbols(r"\sigma_1"),
            "self_coupling_2": symbols(r"\sigma_2"),
            "self_coupling_a": symbols(r"\sigma_a"),
            "unitary_loss_coefficient": symbols(r"\gamma"),
        }
    
    @property
    def equations(self):
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = self.pins
        z = self.z
        return [
            Eq(a0, 1),
            Eq(a2, self.parameter_symbols["self_coupling_1"] * a0 + I * self.parameter_symbols["cross_coupling_1"] * a1),
            Eq(a3, I * self.parameter_symbols["cross_coupling_1"] * a0 + self.parameter_symbols["self_coupling_1"] * a1),
            Eq(a6, self.parameter_symbols["self_coupling_2"] * a4 + I * self.parameter_symbols["cross_coupling_2"] * a5),
            Eq(a7, I * self.parameter_symbols["cross_coupling_2"] * a4 + self.parameter_symbols["self_coupling_2"] * a5),
            Eq(a10, self.parameter_symbols["self_coupling_a"] * a8 + I * self.parameter_symbols["cross_coupling_a"] * a9),
            Eq(a11, I * self.parameter_symbols["cross_coupling_a"] * a8 + self.parameter_symbols["self_coupling_a"] * a9),
            Eq(a5, a3 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["m_2"]),
            Eq(a1, a7 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["m_1"]),
            Eq(a4, a10 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["n_2"]),
            Eq(a8, a2 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["n_1"]),
            Eq(a9, a11 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["p"])
        ]
    

class SymPy_TwoHeaded_ResonantInterferometricCoupler(SymPy_PhotonicCircuit):
    num_pins = 16

    def __init__(self, numeric_parameters=None):
        super().__init__(numeric_parameters=numeric_parameters)
    
    @property
    def parameter_symbols(self):
        return {
            **super().parameter_symbols,
            "m_1": symbols("m_1"),
            "m_2": symbols("m_2"),
            "n_1": symbols("n_1"),
            "n_2": symbols("n_2"),
            "n_3": symbols("n_3"),
            "p_1": symbols("p_1"),
            "p_2": symbols("p_2"),
            "cross_coupling_1": symbols(r"\kappa_1"),
            "cross_coupling_2": symbols(r"\kappa_2"),
            "cross_coupling_a": symbols(r"\kappa_a"),
            "cross_coupling_b": symbols(r"\kappa_b"),
            "self_coupling_1": symbols(r"\sigma_1"),
            "self_coupling_2": symbols(r"\sigma_2"),
            "self_coupling_a": symbols(r"\sigma_a"),
            "self_coupling_b": symbols(r"\sigma_b"),
        }
    
    @property
    def equations(self):
        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15 = self.pins
        z = self.z
        return [
            Eq(a0, 1),
            Eq(a2, self.parameter_symbols["self_coupling_1"] * a0 + I * self.parameter_symbols["cross_coupling_1"] * a1),
            Eq(a3, I * self.parameter_symbols["cross_coupling_1"] * a0 + self.parameter_symbols["self_coupling_1"] * a1),
            Eq(a6, self.parameter_symbols["self_coupling_2"] * a4 + I * self.parameter_symbols["cross_coupling_2"] * a5),
            Eq(a7, I * self.parameter_symbols["cross_coupling_2"] * a4 + self.parameter_symbols["self_coupling_2"] * a5),
            Eq(a10, self.parameter_symbols["self_coupling_a"] * a8 + I * self.parameter_symbols["cross_coupling_a"] * a9),
            Eq(a11, I * self.parameter_symbols["cross_coupling_a"] * a8 + self.parameter_symbols["self_coupling_a"] * a9),
            Eq(a14, self.parameter_symbols["self_coupling_b"] * a12 + I * self.parameter_symbols["cross_coupling_b"] * a13),
            Eq(a15, I * self.parameter_symbols["cross_coupling_b"] * a12 + self.parameter_symbols["self_coupling_b"] * a13),
            Eq(a5, a3 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["m_2"]),
            Eq(a1, a7 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["m_1"]),
            Eq(a4,  a14 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["n_3"]),
            Eq(a12, a10 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["n_2"]),
            Eq(a8,  a2  * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["n_1"]),
            Eq(a9,  a11 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["p_1"]),
            Eq(a13, a15 * (self.parameter_symbols["unitary_loss_coefficient"] * z ** (-1)) ** self.parameter_symbols["p_2"]),
        ]
    
    @property
    def _intrinsic_fwhm(self):
        if not self.numeric_parameters:
            raise ValueError("Numeric parameters must be set before calling intrinsic_fwhm")
        gamma = self.numeric_parameters["unitary_loss_coefficient"]
        m = self.numeric_parameters["m_1"] + self.numeric_parameters["m_2"]
        return 2 * (1 - gamma ** m)