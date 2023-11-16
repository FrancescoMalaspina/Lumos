from .base import CompositeStructure, Pin, wavelength_to_frequency
from .structures import AddDropFilter, Waveguide, RingResonator, Source, AddDropFilterInternalSource
from collections.abc import Sequence
import numpy as np
from scipy.constants import c


class HeadlessSnowman(CompositeStructure):
    num_pins = 12
    num_equations = 12

    def __init__(
            self,
            main_radius: float,
            auxiliary_radius: float,
            mach_zender_length: float,
            input_cross_coupling_coefficient: float,
            through_cross_coupling_coefficient: float,
            ring_cross_coupling_coefficient: float,
            effective_refractive_index: float,
            group_refractive_index: float,
            GVD: float,
            loss_dB: float,  # dB/m
            central_wavelength: float = 1550e-9,
            angular_frequencies: float = wavelength_to_frequency(1550e-9),
            pins: Sequence[Pin] = None):
        Pin.reset_id_iterator()
        super().__init__(effective_refractive_index, group_refractive_index, GVD, loss_dB, central_wavelength,
                         pins, angular_frequencies)
        # lengths
        self.main_radius = main_radius
        self.auxiliary_radius = auxiliary_radius
        self.mach_zender_length = mach_zender_length
        # coupling coefficients
        self.input_cross_coupling_coefficient = input_cross_coupling_coefficient
        self.through_cross_coupling_coefficient = through_cross_coupling_coefficient
        self.ring_cross_coupling_coefficient = ring_cross_coupling_coefficient
        self.structures = [
            Source(pins=[self.pins[0]]),
            AddDropFilter(
                radius=self.main_radius,
                input_cross_coupling_coefficient=self.input_cross_coupling_coefficient,
                auxiliary_cross_coupling_coefficient=self.through_cross_coupling_coefficient,
                effective_refractive_index=self.effective_refractive_index,
                group_refractive_index=self.group_refractive_index,
                GVD=self.GVD,
                loss_dB=self.loss_dB,
                central_wavelength=self.central_wavelength,
                angular_frequencies=self.angular_frequencies,
                pins=self.pins[:8],
            ),
            Waveguide(
                length=self.mach_zender_length / 2,
                effective_refractive_index=self.effective_refractive_index,
                group_refractive_index=self.group_refractive_index,
                GVD=self.GVD,
                loss_dB=self.loss_dB,
                central_wavelength=self.central_wavelength,
                angular_frequencies=self.angular_frequencies,
                pins=[self.pins[2], self.pins[8]],
            ),
            RingResonator(
                radius=self.auxiliary_radius,
                cross_coupling_coefficient=self.ring_cross_coupling_coefficient,
                effective_refractive_index=self.effective_refractive_index,
                group_refractive_index=self.group_refractive_index,
                GVD=self.GVD,
                loss_dB=self.loss_dB,
                central_wavelength=self.central_wavelength,
                angular_frequencies=self.angular_frequencies,
                pins=self.pins[8:],
            ),
            Waveguide(
                length=self.mach_zender_length / 2,
                effective_refractive_index=self.effective_refractive_index,
                group_refractive_index=self.group_refractive_index,
                GVD=self.GVD,
                loss_dB=self.loss_dB,
                central_wavelength=self.central_wavelength,
                angular_frequencies=self.angular_frequencies,
                pins=[self.pins[10], self.pins[4]],
            ),
        ]

    def __str__(self):
        """ Return a string representation of the Pin object. """
        return f"RingInterferometer {self.id}"


class HeadlessSnowmanInternalSource(CompositeStructure):
    num_pins = 12
    num_equations = 12

    def __init__(
            self,
            main_radius: float,
            auxiliary_radius: float,
            mach_zender_length: float,
            input_cross_coupling_coefficient: float,
            through_cross_coupling_coefficient: float,
            ring_cross_coupling_coefficient: float,
            effective_refractive_index: float,
            group_refractive_index: float,
            GVD: float,
            loss_dB: float,  # dB/m
            central_wavelength: float = 1550e-9,
            angular_frequencies: float = wavelength_to_frequency(1550e-9),
            pins: Sequence[Pin] = None):
        Pin.reset_id_iterator()
        super().__init__(effective_refractive_index, group_refractive_index, GVD, loss_dB, central_wavelength,
                         pins, angular_frequencies)
        # lengths
        self.main_radius = main_radius
        self.auxiliary_radius = auxiliary_radius
        self.mach_zender_length = mach_zender_length
        # coupling coefficients
        self.input_cross_coupling_coefficient = input_cross_coupling_coefficient
        self.through_cross_coupling_coefficient = through_cross_coupling_coefficient
        self.ring_cross_coupling_coefficient = ring_cross_coupling_coefficient
        self.structures = [
            Source(pins=[self.pins[0]], amplitude=0),
            AddDropFilterInternalSource(
                source_amplitude=1,
                radius=self.main_radius,
                input_cross_coupling_coefficient=self.input_cross_coupling_coefficient,
                auxiliary_cross_coupling_coefficient=self.through_cross_coupling_coefficient,
                effective_refractive_index=self.effective_refractive_index,
                group_refractive_index=self.group_refractive_index,
                GVD=self.GVD,
                loss_dB=self.loss_dB,
                central_wavelength=self.central_wavelength,
                angular_frequencies=self.angular_frequencies,
                pins=self.pins[:8],
            ),
            Waveguide(
                length=self.mach_zender_length / 2,
                effective_refractive_index=self.effective_refractive_index,
                group_refractive_index=self.group_refractive_index,
                GVD=self.GVD,
                loss_dB=self.loss_dB,
                central_wavelength=self.central_wavelength,
                angular_frequencies=self.angular_frequencies,
                pins=[self.pins[2], self.pins[8]],
            ),
            RingResonator(
                radius=self.auxiliary_radius,
                cross_coupling_coefficient=self.ring_cross_coupling_coefficient,
                effective_refractive_index=self.effective_refractive_index,
                group_refractive_index=self.group_refractive_index,
                GVD=self.GVD,
                loss_dB=self.loss_dB,
                central_wavelength=self.central_wavelength,
                angular_frequencies=self.angular_frequencies,
                pins=self.pins[8:],
            ),
            Waveguide(
                length=self.mach_zender_length / 2,
                effective_refractive_index=self.effective_refractive_index,
                group_refractive_index=self.group_refractive_index,
                GVD=self.GVD,
                loss_dB=self.loss_dB,
                central_wavelength=self.central_wavelength,
                angular_frequencies=self.angular_frequencies,
                pins=[self.pins[10], self.pins[4]],
            ),
        ]
