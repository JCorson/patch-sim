"""Integration tests for CalciumDynamics in end-to-end simulations.

Verifies that ca_i columns appear (or are absent) correctly in
simulate_current_clamp and simulate_voltage_clamp outputs, and that
calcium dynamics behave correctly during full HH simulations.
Unit tests for CalciumDynamics in isolation live in tests/unit/test_calcium.py.
"""

import numpy as np

from patch_sim.calcium import CalciumDynamics
from patch_sim.channels import (
    GatingVariable,
    IonChannel,
    IonSpecies,
    NernstSpec,
)
from patch_sim.clamp_simulations import simulate_current_clamp, simulate_voltage_clamp
from patch_sim.neuron import Neuron
from patch_sim.protocols import step_current, step_voltage

# ---------------------------------------------------------------------------
# Minimal calcium channel shared across tests.
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
# Backward compatibility — no ca_i column without calcium_dynamics
# ---------------------------------------------------------------------------


def test_current_clamp_no_ca_column_by_default(hh_model: Neuron) -> None:
    """simulate_current_clamp returns no ca_i column when calcium_dynamics is None."""
    protocol = step_current(
        duration=5.0,
        current_amplitude=10.0,
        step_start=1.0,
        step_duration=3.0,
    )
    df = simulate_current_clamp(hh_model, protocol)
    assert df.dtype.names is not None
    assert "ca_i" not in df.dtype.names


def test_voltage_clamp_no_ca_column_by_default(hh_model: Neuron) -> None:
    """simulate_voltage_clamp returns no ca_i column when calcium_dynamics is None."""
    protocol = step_voltage(
        duration=5.0,
        voltage_amplitude=0.0,
        step_start=1.0,
        step_duration=3.0,
        holding_voltage=-65.0,
    )
    df = simulate_voltage_clamp(hh_model, protocol)
    assert df.dtype.names is not None
    assert "ca_i" not in df.dtype.names


# ---------------------------------------------------------------------------
# With calcium_dynamics but no calcium channels: ca_i stays at ca_rest
# ---------------------------------------------------------------------------


def test_current_clamp_ca_stays_at_rest_no_calcium_channels() -> None:
    """ca_i stays at ca_rest throughout when no channel carries calcium."""
    cd = CalciumDynamics()
    neuron = Neuron(calcium_dynamics=cd)
    protocol = step_current(
        duration=5.0,
        current_amplitude=10.0,
        step_start=1.0,
        step_duration=3.0,
    )
    df = simulate_current_clamp(neuron, protocol)
    assert df.dtype.names is not None
    assert "ca_i" in df.dtype.names
    np.testing.assert_allclose(np.asarray(df["ca_i"]), cd.ca_rest, rtol=1e-6)


def test_voltage_clamp_ca_stays_at_rest_no_calcium_channels() -> None:
    """ca_i stays at ca_rest throughout when no channel carries calcium."""
    cd = CalciumDynamics()
    neuron = Neuron(calcium_dynamics=cd)
    protocol = step_voltage(
        duration=5.0,
        voltage_amplitude=0.0,
        step_start=1.0,
        step_duration=3.0,
        holding_voltage=-65.0,
    )
    df = simulate_voltage_clamp(neuron, protocol)
    assert df.dtype.names is not None
    assert "ca_i" in df.dtype.names
    np.testing.assert_allclose(np.asarray(df["ca_i"]), cd.ca_rest, rtol=1e-6)


# ---------------------------------------------------------------------------
# carries_calcium=True in simulations: ca_i column exists and varies
# ---------------------------------------------------------------------------


def test_current_clamp_ca_varies_with_calcium_channel() -> None:
    """ca_i column is present and changes from ca_rest when a Ca2+ channel exists."""
    cd = CalciumDynamics(alpha_ca=1e-3, tau_ca=500.0, ca_rest=1e-4)
    neuron = Neuron(
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
    assert df.dtype.names is not None
    assert "ca_i" in df.dtype.names
    # With a calcium-carrying channel that has inward drive, [Ca2+] should change
    assert not np.allclose(np.asarray(df["ca_i"]), cd.ca_rest, rtol=1e-6)


def test_voltage_clamp_ca_varies_with_calcium_channel() -> None:
    """ca_i column is present and changes from ca_rest in voltage clamp."""
    cd = CalciumDynamics(alpha_ca=1e-3, tau_ca=500.0, ca_rest=1e-4)
    neuron = Neuron(
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
    assert df.dtype.names is not None
    assert "ca_i" in df.dtype.names
    assert not np.allclose(np.asarray(df["ca_i"]), cd.ca_rest, rtol=1e-6)


# ---------------------------------------------------------------------------
# ca_i stays non-negative
# ---------------------------------------------------------------------------


def test_ca_i_stays_non_negative_current_clamp() -> None:
    """ca_i is never negative throughout a current-clamp simulation."""
    cd = CalciumDynamics(alpha_ca=1.0, tau_ca=1.0, ca_rest=0.0)
    neuron = Neuron(
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
    neuron = Neuron(
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
