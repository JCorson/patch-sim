"""Ion channel abstractions for the conductance-based neuron simulator.

Provides the building blocks for defining additional ion channels that can be
added on top of the classic Na, K, and leak channels.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

RateFn = Callable[[float, float], float]


class IonSpecies(Enum):
    """Ion species with associated valence and symbol.

    Each member stores its chemical symbol and ionic valence, which are used
    when looking up concentrations from the neuron model and when computing
    Nernst potentials.

    Attributes:
        SODIUM: Na⁺, valence +1.
        POTASSIUM: K⁺, valence +1.
        CALCIUM: Ca²⁺, valence +2.
    """

    SODIUM = ("Na", 1)
    POTASSIUM = ("K", 1)
    CALCIUM = ("Ca", 2)

    def __init__(self, symbol: str, valence: int) -> None:
        """Initialise the enum member with its symbol and valence.

        Args:
            symbol: Chemical symbol string (e.g. ``'Na'``).
            valence: Signed ionic charge (e.g. ``+1`` for Na⁺, ``+2`` for Ca²⁺).
        """
        self._symbol = symbol
        self._valence = valence

    @property
    def symbol(self) -> str:
        """Chemical symbol string (e.g. ``'Na'``)."""
        return self._symbol

    @property
    def valence(self) -> int:
        """Signed ionic valence (e.g. +1 for Na⁺, -1 for Cl⁻)."""
        return self._valence


@dataclass(frozen=True)
class NernstSpec:
    """Reversal potential specification using the Nernst equation.

    The reversal potential is computed at simulation time from the ion
    concentrations stored in the neuron model using the Nernst equation.

    Attributes:
        species: The ion species that carries current through this channel.
    """

    species: IonSpecies


@dataclass(frozen=True)
class GoldmanSpec:
    """Reversal potential specification using the Goldman-Hodgkin-Katz equation.

    Used for channels permeable to multiple monovalent ion species (e.g. Ih,
    which passes both Na⁺ and K⁺).  Each entry in *permeabilities* provides a
    relative permeability weight and the corresponding :class:`IonSpecies`; the
    actual concentrations are looked up from the neuron model at simulation time.

    All species listed must be monovalent (|valence| == 1).  For divalent ions
    use :class:`NernstSpec` instead.

    Attributes:
        permeabilities: Tuple of ``(species, relative_permeability)`` pairs.
            Permeabilities are dimensionless and relative to each other; their
            absolute scale cancels in the GHK equation.

    Raises:
        ValueError: If any species has |valence| != 1 or if any relative
            permeability is negative.
    """

    permeabilities: tuple[tuple[IonSpecies, float], ...]

    def __post_init__(self) -> None:
        """Validate that all species are monovalent and permeabilities non-negative."""
        for species, p in self.permeabilities:
            if abs(species.valence) != 1:
                raise ValueError(
                    f"GoldmanSpec only supports monovalent ions; "
                    f"{species.symbol} has valence {species.valence}. "
                    "Use NernstSpec for divalent ions."
                )
            if p < 0:
                raise ValueError(
                    f"Relative permeability for {species.symbol} must be "
                    f"non-negative, got {p}."
                )


# Union type for reversal potential specifications.
ReversalSpec = NernstSpec | GoldmanSpec


@dataclass(frozen=True)
class GatingVariable:
    """A single gating variable with its kinetic rate functions.

    Rate functions always accept ``(V, ca_i)`` — voltage in mV and
    intracellular Ca²⁺ concentration in mM.  Voltage-only gates simply ignore
    the ``ca_i`` argument.

    Attributes:
        name: Unique name used as a key in gating-state dicts (e.g. 'r').
        power: Exponent applied to this gate's value when computing conductance.
        alpha: Forward rate function ``alpha(V, ca_i)`` in units 1/ms.
        beta: Backward rate function ``beta(V, ca_i)`` in units 1/ms.
    """

    name: str
    power: int
    alpha: RateFn
    beta: RateFn


@dataclass(frozen=True)
class IonChannel:
    """An ion channel whose reversal potential is derived from ion concentrations.

    Computes current as ``g_max * prod(gate^power) * (V - E_rev)`` where
    ``E_rev`` is computed at runtime from the neuron's ion concentrations using
    either the Nernst equation (:class:`NernstSpec`) or the Goldman-Hodgkin-Katz
    equation (:class:`GoldmanSpec`).

    Attributes:
        name: Human-readable channel identifier (e.g. ``'Ih'``).
        g_max: Maximum conductance in mS/cm².
        gating_variables: Tuple of gating variable descriptors.
        reversal_spec: Specification for how the reversal potential is computed
            from the neuron's ion concentrations.
        carries_calcium: ``True`` for channels that carry Ca²⁺ ions (e.g.
            ICaL, ICaT, ICaN).  Used by
            :meth:`~patch_sim.neuron.Neuron.calcium_current` to
            sum Ca²⁺ influx for the intracellular Ca²⁺ ODE.  Defaults to
            ``False``.

    Raises:
        ValueError: If ``g_max`` is negative or if gating variable names are
            not unique within the channel.
    """

    name: str
    g_max: float
    gating_variables: tuple["GatingVariable", ...]
    reversal_spec: ReversalSpec
    carries_calcium: bool = field(default=False)

    def __post_init__(self) -> None:
        """Validate channel parameters on construction."""
        if self.g_max < 0:
            raise ValueError(
                f"Channel '{self.name}': g_max must be non-negative, got {self.g_max}."
            )
        names = [gv.name for gv in self.gating_variables]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Channel '{self.name}': gating variable names must be unique, "
                f"got {names}."
            )

    @property
    def current_name(self) -> str:
        """Structured-array field name for this channel's current contribution.

        Always prepends ``'I'`` to the channel name, so e.g. a channel named
        ``'Na'`` contributes a field named ``'INa'``, and a channel named
        ``'h'`` contributes a field named ``'Ih'``.

        Returns:
            The field name string used in simulation result structured arrays.
        """
        return f"I{self.name}"

    def reversal_potential(self, neuron: Any) -> float:
        """Compute the reversal potential from the neuron's ion concentrations.

        Dispatches to :func:`~patch_sim.electrochemistry.nernst_potential` for a
        :class:`NernstSpec` or
        :func:`~patch_sim.electrochemistry.goldman_potential` for a
        :class:`GoldmanSpec`, using the concentrations stored on *neuron*.

        Args:
            neuron: A :class:`~patch_sim.neuron.Neuron` instance
                whose ``ion_concentrations()`` method is used to look up ion
                concentrations.

        Returns:
            Reversal potential in mV.
        """
        from .electrochemistry import goldman_potential, nernst_potential

        spec = self.reversal_spec
        if isinstance(spec, NernstSpec):
            c_out, c_in = neuron.ion_concentrations(spec.species)
            return nernst_potential(spec.species.valence, neuron.T, c_out, c_in)
        # GoldmanSpec — separate cations from anions
        cation_terms = []
        anion_terms = []
        for species, p in spec.permeabilities:
            c_out, c_in = neuron.ion_concentrations(species)
            if species.valence > 0:
                cation_terms.append((p, c_out, c_in))
            else:
                anion_terms.append((p, c_out, c_in))
        return goldman_potential(neuron.T, cation_terms, anion_terms)

    def compute_current(
        self,
        V: float,
        gating_state: dict[str, float],
        neuron: Any,
        e_rev: float | None = None,
    ) -> float:
        """Compute the ionic current through this channel.

        Evaluates ``g_max * prod(gate^power) * (V - E_rev)``.  When *e_rev* is
        supplied it is used directly, bypassing :meth:`reversal_potential`.  The
        simulation core passes a pre-computed value here to avoid recomputing
        the Nernst/Goldman log on every RK4 substep.

        Args:
            V: Membrane voltage in mV.
            gating_state: Mapping from gating variable name to current value.
            neuron: The :class:`~patch_sim.neuron.Neuron` model used to compute
                the reversal potential when *e_rev* is not supplied.
            e_rev: Pre-computed reversal potential in mV.  When ``None``
                (default), :meth:`reversal_potential` is called instead.

        Returns:
            Ionic current in µA/cm².
        """
        g = self.g_max
        for gv in self.gating_variables:
            g *= gating_state[gv.name] ** gv.power
        if e_rev is None:
            e_rev = self.reversal_potential(neuron)
        return g * (V - e_rev)
