from .base import CompositeStructure, Pin, wavelength_to_frequency
from .structures import AddDropFilter, Waveguide, RingResonator, Source
from collections.abc import Sequence
import numpy as np
import matplotlib.pyplot as plt
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
    
    @property
    def transmission(self, pin_id=6):
        return np.power(np.abs(self.fields[:, pin_id]), 2)

    def plot_spectrum(self):
        plt.figure(figsize=(12, 7))
        # IE = self.intensity_enhancement
        FE = self.field_enhancement
        plt.plot(self.angular_frequencies, FE, label="simulated", linewidth=1)
        omega_0 = wavelength_to_frequency(1550e-9)
        plt.vlines(omega_0, 0, np.max(FE), label="omega_0", linestyle="--", linewidth=2, color="grey")
        plt.xlabel("Angular frequency [s-1]")
        plt.ylabel("Field enhancement")
        plt.title("Field Enhancement spectrum @ Pin 1")
        # plt.yscale("log")
        plt.legend()
        plt.grid()
        # plt.show()

    def get_spectrum(self):
        FE = self.field_enhancement
        angular_frequencies = self.angular_frequencies
        return angular_frequencies, FE


def headless_snowman_debug():
    omega_0 = wavelength_to_frequency(1550e-9)
    omega_m = c / 120e-6
    angular_frequencies = np.linspace(omega_0 - 1.5 * omega_m, omega_0 + 1.5 * omega_m, 5000)
    hs = HeadlessSnowman(
        main_radius=120e-6,
        auxiliary_radius=90e-6,
        mach_zender_length=np.pi * 120e-6,
        input_cross_coupling_coefficient=0.1,
        through_cross_coupling_coefficient=0.1,
        ring_cross_coupling_coefficient=0,
        effective_refractive_index=1.7,
        group_refractive_index=2,
        GVD=0 - 6e-24,
        loss_dB=10,
        central_wavelength=1550e-9,
        angular_frequencies=angular_frequencies,
    )
    # hs.plot_spectrum()
    print(hs.get_spectrum())


def headless_snowman_with_active_ring_debug():
    omega_0 = wavelength_to_frequency(1550e-9)
    omega_m = c / 120e-6
    angular_frequencies = np.linspace(omega_0 - 1.5 * omega_m, omega_0 + 1.5 * omega_m, 50000)
    hs = HeadlessSnowman(
        main_radius=120e-6,
        auxiliary_radius=90e-6,
        mach_zender_length=3 * np.pi * 120e-6,
        input_cross_coupling_coefficient=0.05,
        through_cross_coupling_coefficient=0.05,
        ring_cross_coupling_coefficient=0.3,
        effective_refractive_index=1.7,
        group_refractive_index=2,
        GVD=0 - 6e-24,
        loss_dB=10,
        central_wavelength=1550e-9,
        angular_frequencies=angular_frequencies,
    )
    hs.plot_spectrum()
    # plt.yscale("log")
    # angular_frequencies, FE = hs.get_spectrum()
    # np.save("FE_1pi", FE)
    # np.save("angular_frequencies", angular_frequencies)


if __name__ == "__main__":
    import time

    start_time = time.time()

    # headless_snowman_debug()
    headless_snowman_with_active_ring_debug()

    end_time = time.time()

    print("Execution time:", end_time - start_time, "seconds")

    plt.show()
    # time with matrix inversion: 15.025897979736328 seconds
    # time with np.linalg.solve: 14.259375810623169 seconds
