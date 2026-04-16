"""Unit tests for patch_sim.calcium and IonChannel.carries_calcium.

Covers CalciumDynamics ODE correctness, validation, the carries_calcium flag
on IonChannel, and public API export. Simulation integration tests live in
tests/integration/test_calcium_simulation.py.
"""

import pytest

import patch_sim
from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    GatingVariable,
    IonChannel,
    IonSpecies,
    NernstSpec,
)

# ---------------------------------------------------------------------------
# Minimal calcium channel for use in tests.
# ---------------------------------------------------------------------------

_CA_GATE = GatingVariable(
    name="ca_gate",
    power=1,
    alpha=lambda V, ca_i: 0.1,
    beta=lambda V, ca_i: 0.1,
)

_MOCK_CALCIUM_CHANNEL = IonChannel(
    name="mock_ca",
    g_max=1.0,
    gating_variables=(_CA_GATE,),
    reversal_spec=NernstSpec(IonSpecies.CALCIUM),
    carries_calcium=True,
)

# ---------------------------------------------------------------------------
# CalciumDynamics.derivative
# ---------------------------------------------------------------------------


def test_derivative_at_rest_is_zero() -> None:
    """d[Ca2+]/dt is zero when ca_i equals ca_rest and I_Ca is zero."""
    cd = CalciumDynamics()
    d = cd.derivative(I_Ca=0.0, ca_i=cd.ca_rest)
    assert d == pytest.approx(0.0)


def test_derivative_inward_current_increases_ca() -> None:
    """Inward Ca2+ current (negative I_Ca) raises intracellular Ca2+."""
    cd = CalciumDynamics(alpha_ca=1e-4, tau_ca=200.0, ca_rest=1e-4)
    # At rest with an inward current: only the -alpha*I_Ca term contributes
    d = cd.derivative(I_Ca=-1.0, ca_i=cd.ca_rest)
    assert d > 0.0
    assert d == pytest.approx(1e-4)


def test_derivative_decay_toward_rest() -> None:
    """When I_Ca=0 and ca_i > ca_rest, the derivative is negative (decay)."""
    cd = CalciumDynamics(alpha_ca=1e-4, tau_ca=200.0, ca_rest=1e-4)
    elevated_ca = 1e-3
    d = cd.derivative(I_Ca=0.0, ca_i=elevated_ca)
    assert d < 0.0
    expected = -(elevated_ca - cd.ca_rest) / cd.tau_ca
    assert d == pytest.approx(expected)


def test_derivative_outward_current_decreases_ca() -> None:
    """Outward Ca2+ current (positive I_Ca) lowers intracellular Ca2+."""
    cd = CalciumDynamics(alpha_ca=1e-4, tau_ca=200.0, ca_rest=1e-4)
    d = cd.derivative(I_Ca=1.0, ca_i=cd.ca_rest)
    assert d < 0.0


# ---------------------------------------------------------------------------
# CalciumDynamics validation
# ---------------------------------------------------------------------------


def test_validation_alpha_ca_zero_raises() -> None:
    """alpha_ca=0 should raise ValueError."""
    with pytest.raises(ValueError, match="alpha_ca"):
        CalciumDynamics(alpha_ca=0.0)


def test_validation_alpha_ca_negative_raises() -> None:
    """Negative alpha_ca should raise ValueError."""
    with pytest.raises(ValueError, match="alpha_ca"):
        CalciumDynamics(alpha_ca=-1e-4)


def test_validation_tau_ca_zero_raises() -> None:
    """tau_ca=0 should raise ValueError."""
    with pytest.raises(ValueError, match="tau_ca"):
        CalciumDynamics(tau_ca=0.0)


def test_validation_tau_ca_negative_raises() -> None:
    """Negative tau_ca should raise ValueError."""
    with pytest.raises(ValueError, match="tau_ca"):
        CalciumDynamics(tau_ca=-200.0)


def test_validation_ca_rest_negative_raises() -> None:
    """Negative ca_rest should raise ValueError."""
    with pytest.raises(ValueError, match="ca_rest"):
        CalciumDynamics(ca_rest=-1e-4)


def test_validation_ca_rest_zero_is_valid() -> None:
    """ca_rest=0 is allowed (fully buffered scenario)."""
    cd = CalciumDynamics(ca_rest=0.0)
    assert cd.ca_rest == 0.0


# ---------------------------------------------------------------------------
# carries_calcium flag
# ---------------------------------------------------------------------------


def test_plain_ion_channel_does_not_carry_calcium() -> None:
    """An IonChannel created without carries_calcium has carries_calcium=False."""
    gv = GatingVariable(
        name="x", power=1, alpha=lambda V, ca_i: 0.01, beta=lambda V, ca_i: 0.01
    )
    ch = IonChannel(
        name="test_ch",
        g_max=1.0,
        gating_variables=(gv,),
        reversal_spec=NernstSpec(IonSpecies.POTASSIUM),
    )
    assert not ch.carries_calcium


def test_mock_calcium_channel_carries_calcium() -> None:
    """An IonChannel with carries_calcium=True has carries_calcium=True."""
    assert _MOCK_CALCIUM_CHANNEL.carries_calcium


# ---------------------------------------------------------------------------
# Public API export
# ---------------------------------------------------------------------------


def test_calcium_dynamics_exported_from_patch_sim() -> None:
    """CalciumDynamics is accessible from the top-level patch_sim package."""
    assert hasattr(patch_sim, "CalciumDynamics")
    cd = patch_sim.CalciumDynamics()
    assert isinstance(cd, CalciumDynamics)
