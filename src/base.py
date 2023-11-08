from abc import ABC, abstractmethod
from itertools import count
from collections.abc import Sequence
import numpy as np
from scipy.constants import c


class Pin:
    """ Class for the definition of pins, which are the input and output ports
    of photonic structures. """
    id_iterator = count()
    unused_ids = []
    num_pins_created = 0

    def __init__(self, label: str = "label"):
        """ Each pin of each structure should have a unique id. """
        if Pin.unused_ids:
            self.id = Pin.unused_ids.pop(0)
        else:
            self.id = next(Pin.id_iterator)
        Pin.num_pins_created += 1
        self.label = label

    def __str__(self):
        """ Return a string representation of the Pin object. """
        return f"Pin {self.id}"

    def delete(self):
        """ Delete this pin and add its id to the list of unused ids. """
        Pin.unused_ids.append(self.id)
        Pin.num_pins_created -= 1

    @classmethod
    def get_num_pins_created(cls):
        """ Get the number of Pin objects that have been created. """
        return cls.num_pins_created

    @classmethod
    def reset_id_iterator(cls):
        cls.id_iterator = count()


class BaseStructure(ABC):
    """ Abstract base class, the fundamental structures (Waveguide, DirectionalCoupler, Souce),
    i.e. those whose equations cannot be derived from other objects. """
    id_iterator = count()
    num_pins = 2  # default number of pins for a structure, overload this in the subclasses
    num_equations = 1

    def __init__(
            self,
            pins: Sequence[Pin] = None,
    ):
        """ Initialize the class. """
        self.id = next(BaseStructure.id_iterator)

        # pins initialization
        if pins is None:
            self.pins = [Pin(self) for _ in range(self.num_pins)]
        elif len(pins) == self.num_pins:
            self.pins = pins
        else:
            print(f"Error: provided {len(pins)} Pins, but the structure: {self} requires {self.num_pins} pins.")
            raise ValueError

    def __str__(self):
        """ Return a string representation of the Pin object. """
        return f"BaseStructure {self.id}"

    @property
    @abstractmethod
    def field_equations(self):
        """ Return the field equations for the structure. """
        pass

    @property
    def ordinate_vector(self):
        """ Return the ordinate vector for the structure. """
        return [0] * self.num_equations


class CompositeStructure(BaseStructure):
    """ This class inherits from PhotonicBaseStructure, it serves to model complex structures, composed
    of many fundanmental structures (DirectionalCoupler, Waveguide, Source)
    """
    num_pins = 2  # default number of pins for a structure, overload this in the subclasses

    def __init__(
            self,
            effective_refractive_index: float = 1,
            group_refractive_index: float = 1,
            GVD: float = 0,
            loss_dB: float = 10,  # dB/m
            central_wavelength: float = 1550e-9,
            pins: Sequence[Pin] = None,
            angular_frequencies: Sequence[float] = None,
            structures: Sequence[BaseStructure] = None,
    ):
        """ Initialize the class. """
        super().__init__(pins=pins)
        self.effective_refractive_index = effective_refractive_index
        self.group_refractive_index = group_refractive_index
        self.GVD = GVD
        self.loss_dB = loss_dB
        self.central_wavelength = central_wavelength
        # frequencies initialization
        try:
            self.angular_frequencies = np.array(angular_frequencies)  # or np.zeros(1)
        except TypeError:
            raise TypeError(f"In {self} angular_frequencies must be a sequence of floats.")
        self.structures = structures or []

    @property
    def wavevector(self):
        """ Return the wavevector for the structure, expanded to the second order in the angular frequency."""
        central_frequency = wavelength_to_frequency(self.central_wavelength)
        zero_order_term = self.effective_refractive_index * central_frequency / c
        first_order_term = self.group_refractive_index * (self.angular_frequencies - central_frequency) / c
        second_order_term = 1 / 2 * self.GVD * np.power((self.angular_frequencies - central_frequency), 2)
        return zero_order_term + first_order_term + second_order_term

    @property
    def field_equations(self):
        """ Return the field equations for the structure. """
        equations = []
        for structure in self.structures:
            equations.extend(structure.field_equations)
        return equations

    @property
    def ordinate_vector(self):
        """ Return the ordinate vector for the structure. """
        ordinate_vector = []  # 1j * np.zeros((self.num_pins))
        for structure in self.structures:
            ordinate_vector.extend(structure.ordinate_vector)
        return ordinate_vector

    # TODO: these methods should be moved to the PhotonicCircuit class, where a Source object should be added to the
    #  structure sequence
    @property
    def coefficient_matrix(self):
        """ Return the coefficient matrix of the system for the structure. To parallelize the computation at 
        different frequencies, the coefficient matrix is an array of dimensions: (num_frequencies, num_pins, num_pins). """
        num_frequencies = len(self.angular_frequencies)
        coefficient_matrix = 1j * np.zeros((num_frequencies, self.num_pins, self.num_pins))
        for i, equation in enumerate(self.field_equations):
            for pin, coefficients in equation.items():
                coefficient_matrix[:, i, pin.id] += coefficients
        return coefficient_matrix

    @property
    def fields(self):
        ordinate_vector = np.broadcast_to(np.array(self.ordinate_vector),
                                          (len(self.angular_frequencies), len(self.ordinate_vector)))
        fields = np.linalg.solve(self.coefficient_matrix, ordinate_vector)
        # fields = np.linalg.inv(self.coefficient_matrix) @ self.ordinate_vector
        return fields

    @property
    def field_enhancement(self, pin_id=1):
        return np.abs(self.fields[:, pin_id])

    @property
    def intensity_enhancement(self, pin_id=1):
        return np.power(np.abs(self.fields[:, pin_id]), 2)

    @property
    def transmission(self, pin_id=2):
        return np.power(np.abs(self.fields[:, pin_id]), 2)

    def __str__(self):
        """ Return a string representation of the Pin object. """
        return f"Structure {self.id}"


def wavelength_to_frequency(wavelength):
    return 2 * np.pi / wavelength * c


def frequency_to_wavelength(omega):
    return 2 * np.pi / omega * c


if __name__ == "__main__":
    structureSequence = [CompositeStructure(), CompositeStructure()]
    print(structureSequence)
