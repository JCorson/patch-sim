"""Tests for intracellular Ca2+ dynamics (issue #44).

Covers CalciumDynamics ODE correctness, validation, backward compatibility,
simulation integration, and the carries_calcium flag on IonChannel.
"""

import numpy as np
import pytest

import patch_sim
from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    GatingVariable,
    IonChannel,
    IonSpecies,
    NernstSpec,
)
from patch_sim.clamp_simulations import simulate_current_clamp, simulate_voltage_clamp
from patch_sim.hodgkin_huxley import HodgkinHuxley
from patch_sim.protocols import step_current, step_voltage

# ---------------------------------------------------------------------------
# Minimal calcium channel for use in tests.
# Uses IonChannel with carries_calcium=True.
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
# Backward compatibility — no ca_i column without calcium_dynamics
# ---------------------------------------------------------------------------


def test_current_clamp_no_ca_column_by_default(hh_model: HodgkinHuxley) -> None:
    """simulate_current_clamp returns no ca_i column when calcium_dynamics is None."""
    protocol = step_current(
        duration=5.0,
        current_amplitude=10.0,
        step_start=1.0,
        step_duration=3.0,
    )
    df = simulate_current_clamp(hh_model, protocol)
    assert "ca_i" not in df.columns


def test_voltage_clamp_no_ca_column_by_default(hh_model: HodgkinHuxley) -> None:
    """simulate_voltage_clamp returns no ca_i column when calcium_dynamics is None."""
    protocol = step_voltage(
        duration=5.0,
        voltage_amplitude=0.0,
        step_start=1.0,
        step_duration=3.0,
        holding_voltage=-65.0,
    )
    df = simulate_voltage_clamp(hh_model, protocol)
    assert "ca_i" not in df.columns


# ---------------------------------------------------------------------------
# With calcium_dynamics but no calcium channels: ca_i stays at ca_rest
# ---------------------------------------------------------------------------


def test_current_clamp_ca_stays_at_rest_no_calcium_channels() -> None:
    """ca_i stays at ca_rest throughout when no channel carries calcium."""
    cd = CalciumDynamics()
    neuron = HodgkinHuxley(calcium_dynamics=cd)
    protocol = step_current(
        duration=5.0,
        current_amplitude=10.0,
        step_start=1.0,
        step_duration=3.0,
    )
    df = simulate_current_clamp(neuron, protocol)
    assert "ca_i" in df.columns
    np.testing.assert_allclose(np.asarray(df["ca_i"]), cd.ca_rest, rtol=1e-6)


def test_voltage_clamp_ca_stays_at_rest_no_calcium_channels() -> None:
    """ca_i stays at ca_rest throughout when no channel carries calcium."""
    cd = CalciumDynamics()
    neuron = HodgkinHuxley(calcium_dynamics=cd)
    protocol = step_voltage(
        duration=5.0,
        voltage_amplitude=0.0,
        step_start=1.0,
        step_duration=3.0,
        holding_voltage=-65.0,
    )
    df = simulate_voltage_clamp(neuron, protocol)
    assert "ca_i" in df.columns
    np.testing.assert_allclose(np.asarray(df["ca_i"]), cd.ca_rest, rtol=1e-6)


# ---------------------------------------------------------------------------
# carries_calcium=True in simulations: ca_i column exists and varies
# ---------------------------------------------------------------------------


def test_current_clamp_ca_varies_with_calcium_channel() -> None:
    """ca_i column is present and changes from ca_rest when a Ca2+ channel exists."""
    cd = CalciumDynamics(alpha_ca=1e-3, tau_ca=500.0, ca_rest=1e-4)
    neuron = HodgkinHuxley(
        calcium_dynamics=cd,
        additional_channels=(_MOCK_CALCIUM_CHANNEL,),
    )
    protocol = step_current(
        duration=20.0,
        current_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
    )
    df = simulate_current_clamp(neuron, protocol)
    assert "ca_i" in df.columns
    # With a calcium-carrying channel that has inward drive, [Ca2+] should change
    assert not np.allclose(np.asarray(df["ca_i"]), cd.ca_rest, rtol=1e-6)


def test_voltage_clamp_ca_varies_with_calcium_channel() -> None:
    """ca_i column is present and changes from ca_rest in voltage clamp."""
    cd = CalciumDynamics(alpha_ca=1e-3, tau_ca=500.0, ca_rest=1e-4)
    neuron = HodgkinHuxley(
        calcium_dynamics=cd,
        additional_channels=(_MOCK_CALCIUM_CHANNEL,),
    )
    protocol = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
        holding_voltage=-65.0,
    )
    df = simulate_voltage_clamp(neuron, protocol)
    assert "ca_i" in df.columns
    assert not np.allclose(np.asarray(df["ca_i"]), cd.ca_rest, rtol=1e-6)


# ---------------------------------------------------------------------------
# ca_i stays non-negative
# ---------------------------------------------------------------------------


def test_ca_i_stays_non_negative_current_clamp() -> None:
    """ca_i is never negative throughout a current-clamp simulation."""
    cd = CalciumDynamics(alpha_ca=1.0, tau_ca=1.0, ca_rest=0.0)
    neuron = HodgkinHuxley(
        calcium_dynamics=cd,
        additional_channels=(_MOCK_CALCIUM_CHANNEL,),
    )
    protocol = step_current(
        duration=5.0,
        current_amplitude=0.0,
        step_start=1.0,
        step_duration=3.0,
    )
    df = simulate_current_clamp(neuron, protocol)
    assert bool(np.all(np.asarray(df["ca_i"]) >= 0.0))


def test_ca_i_stays_non_negative_voltage_clamp() -> None:
    """ca_i is never negative throughout a voltage-clamp simulation."""
    cd = CalciumDynamics(alpha_ca=1.0, tau_ca=1.0, ca_rest=0.0)
    neuron = HodgkinHuxley(
        calcium_dynamics=cd,
        additional_channels=(_MOCK_CALCIUM_CHANNEL,),
    )
    protocol = step_voltage(
        duration=5.0,
        voltage_amplitude=0.0,
        step_start=1.0,
        step_duration=3.0,
        holding_voltage=-65.0,
    )
    df = simulate_voltage_clamp(neuron, protocol)
    assert bool(np.all(np.asarray(df["ca_i"]) >= 0.0))


# ---------------------------------------------------------------------------
# Public API export
# ---------------------------------------------------------------------------


def test_calcium_dynamics_exported_from_patch_sim() -> None:
    """CalciumDynamics is accessible from the top-level patch_sim package."""
    assert hasattr(patch_sim, "CalciumDynamics")
    cd = patch_sim.CalciumDynamics()
    assert isinstance(cd, CalciumDynamics)
