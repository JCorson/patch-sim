"""
This module implements the Hodgkin-Huxley model for simulating action potentials.
The model includes equations for ion channel dynamics and membrane voltage.
"""

from dataclasses import dataclass, field
from .nernst import nernst_potential
from .utils import safe_exp


@dataclass(frozen=True)
class HodgkinHuxley:
    """
    Simulates the Hodgkin-Huxley model of action potentials.

    All parameters are fixed at construction time. The class is immutable
    (frozen) to prevent accidental mutation of parameters after the reversal
    potentials have been computed.

    Attributes:
        C_m (float): Membrane capacitance in uF/cm^2.
        g_Na (float): Maximum sodium conductance in mS/cm^2.
        g_K (float): Maximum potassium conductance in mS/cm^2.
        g_L (float): Leak conductance in mS/cm^2.
        v_rest (float): Resting potential in mV.
        Na_out (float): Extracellular sodium concentration in mM.
        Na_in (float): Intracellular sodium concentration in mM.
        K_out (float): Extracellular potassium concentration in mM.
        K_in (float): Intracellular potassium concentration in mM.
        Cl_out (float): Extracellular chloride concentration in mM.
        Cl_in (float): Intracellular chloride concentration in mM.
        T (float): Temperature in Kelvin.

    Computed properties (derived from ion concentrations at construction time):
        E_Na (float): Sodium reversal potential in mV.
        E_K (float): Potassium reversal potential in mV.
        E_L (float): Leak reversal potential in mV.
    """

    # Membrane properties
    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 0.3
    C_m: float = 1.0
    v_rest: float = -65.0

    # Ion concentrations (in mM)
    Na_out: float = 145.0
    Na_in: float = 15.0
    K_out: float = 5.0
    K_in: float = 140.0
    Cl_out: float = 120.0
    Cl_in: float = 10.0

    # Temperature in Kelvin (37°C for mammalian cells)
    T: float = 310.15

    # Reversal potentials — computed from ion concentrations in __post_init__
    # and stored as regular fields. Use field(init=False) so they are not
    # accepted as constructor arguments.
    E_Na: float = field(init=False)
    E_K: float = field(init=False)
    E_L: float = field(init=False)

    def __post_init__(self) -> None:
        if self.g_Na < 0:
            raise ValueError("Sodium conductance (g_Na) must be non-negative.")
        if self.g_K < 0:
            raise ValueError("Potassium conductance (g_K) must be non-negative.")
        if self.g_L < 0:
            raise ValueError("Leak conductance (g_L) must be non-negative.")
        if self.C_m <= 0:
            raise ValueError("Membrane capacitance (C_m) must be positive.")
        if self.T <= 0:
            raise ValueError("Temperature (T) must be positive (in Kelvin).")
        for name, value in [
            ("Na_out", self.Na_out),
            ("Na_in", self.Na_in),
            ("K_out", self.K_out),
            ("K_in", self.K_in),
            ("Cl_out", self.Cl_out),
            ("Cl_in", self.Cl_in),
        ]:
            if value <= 0:
                raise ValueError(f"Ion concentration ({name}) must be positive.")

        # frozen=True prevents normal attribute assignment, so use object.__setattr__
        object.__setattr__(
            self, "E_Na", nernst_potential(1, self.T, self.Na_out, self.Na_in)
        )
        object.__setattr__(
            self, "E_K", nernst_potential(1, self.T, self.K_out, self.K_in)
        )
        object.__setattr__(
            self, "E_L", nernst_potential(-1, self.T, self.Cl_out, self.Cl_in)
        )

    def alpha_n(self, V: float) -> float:
        """
        Calculate the rate constant alpha_n for potassium channel activation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_n.
        """
        if abs(V + 55) < 1e-6:
            # Handle near-singularity case
            # This is the limit as V approaches -55
            return 0.1
        else:
            denominator = 1 - safe_exp(-(V + 55) / 10)
            return 0.01 * (V + 55) / denominator

    def beta_n(self, V: float) -> float:
        """
        Calculate the rate constant beta_n for potassium channel deactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_n.
        """
        return 0.125 * safe_exp(-(V + 65) / 80)

    def alpha_m(self, V: float) -> float:
        """
        Calculate the rate constant alpha_m for sodium channel activation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_m.
        """
        if abs(V + 40) < 1e-6:
            # Handle near-singularity case
            # This is the limit as V approaches -40
            return 1.0
        else:
            denominator = 1 - safe_exp(-(V + 40) / 10)
            return 0.1 * (V + 40) / denominator

    def beta_m(self, V: float) -> float:
        """
        Calculate the rate constant beta_m for sodium channel deactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_m.
        """
        return 4.0 * safe_exp(-(V + 65) / 18)

    def alpha_h(self, V: float) -> float:
        """
        Calculate the rate constant alpha_h for sodium channel inactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant alpha_h.
        """
        return 0.07 * safe_exp(-(V + 65) / 20)

    def beta_h(self, V: float) -> float:
        """
        Calculate the rate constant beta_h for sodium channel reactivation.

        Parameters:
            V (float): Membrane voltage in mV.

        Returns:
            float: The rate constant beta_h.
        """
        return 1 / (1 + safe_exp(-(V + 35) / 10))
