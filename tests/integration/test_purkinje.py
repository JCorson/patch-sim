"""Behavioral tests for the Purkinje neuron preset.

Verifies that NaP/NaR-driven pacemaking behaviour is present: INaP provides
inward subthreshold current near the pacemaker threshold, INaR is represented
in the simulation output, and the cell fires readily from the unstable
zero-current equilibrium (v_rest = −65 mV).
"""

import numpy as np
import pytest

from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import PURKINJE
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current

# ---------------------------------------------------------------------------
# Shared fixture and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def pk_neuron() -> Neuron:
    """Purkinje neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[PURKINJE])


def _count_action_potentials(voltage: np.ndarray, threshold: float = 0.0) -> int:
    """Count upward threshold crossings in a voltage trace.

    Args:
        voltage: 1-D array of membrane voltages in mV.
        threshold: Voltage threshold for AP detection in mV.

    Returns:
        Number of action potentials detected.
    """
    count = 0
    above = False
    for v in voltage:
        if v > threshold and not above:
            count += 1
            above = True
        elif v < threshold:
            above = False
    return count


def _ms_to_samples(ms: float) -> int:
    """Convert a duration in milliseconds to a sample count.

    Args:
        ms: Duration in milliseconds.

    Returns:
        Corresponding number of simulation samples.
    """
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# INaP: subthreshold inward current
# ---------------------------------------------------------------------------


def test_inap_inward_at_subthreshold(pk_neuron: Neuron) -> None:
    """INaP carries inward (negative) current during subthreshold depolarization.

    The persistent sodium current activates near v_rest = −65 mV and amplifies
    subthreshold depolarizations.  With a 0.05 µA/cm² step (below AP threshold,
    verified to be subthreshold in a 50 ms window), the INaP column must be
    negative throughout the step, confirming the channel is active and
    contributing inward current.
    Ref: Raman & Bean (1999), J. Neurosci. 19:4663.
    """
    protocol = step_current(
        duration=100.0,
        current_amplitude=0.05,
        step_start=0.0,
        step_duration=50.0,
    )
    result = simulate_current_clamp(pk_neuron, current_external=protocol)

    step_end = _ms_to_samples(50)
    peak_voltage = float(result["voltage"][:step_end].max())
    assert peak_voltage < 0.0, (
        f"Step should be subthreshold (< 0 mV), "
        f"but voltage reached {peak_voltage:.1f} mV"
    )

    inap_during_step = result["INaP"][:step_end]
    assert float(inap_during_step.max()) < 0.0, (
        f"Expected INaP to be inward (negative) during depolarization, "
        f"but max was {inap_during_step.max():.4f} µA/cm²"
    )


# ---------------------------------------------------------------------------
# INaR: resurgent Na⁺ channel presence
# ---------------------------------------------------------------------------


def test_inar_column_present_in_simulation(pk_neuron: Neuron) -> None:
    """INaR current column appears in the simulation result dtype.

    Confirms that the INaR channel is wired into the preset and that the
    resurgent sodium current is tracked in the output structured array.
    Ref: Raman & Bean (1997), Neuron 19:1of.
    """
    protocol = step_current(duration=50.0, current_amplitude=5.0)
    result = simulate_current_clamp(pk_neuron, current_external=protocol)
    assert "INaR" in result.dtype.names, (
        "Expected 'INaR' column in simulation output — make_inar_channel may "
        "not be included in the Purkinje preset channels tuple"
    )


# ---------------------------------------------------------------------------
# Near-threshold firing
# ---------------------------------------------------------------------------


def test_fires_from_small_depolarization(pk_neuron: Neuron) -> None:
    """Cell fires at least one AP with a small depolarizing current from v_rest.

    v_rest = −65 mV is the zero-current pacemaker threshold (unstable
    equilibrium).  A 0.2 µA/cm² step — just above the subthreshold range —
    must cross the Na⁺ activation threshold and fire an action potential,
    demonstrating that the cell is at the edge of the pacemaking zone.
    """
    protocol = step_current(
        duration=200.0,
        current_amplitude=0.2,
        step_start=0.0,
        step_duration=100.0,
    )
    result = simulate_current_clamp(pk_neuron, current_external=protocol)
    n_aps = _count_action_potentials(result["voltage"])
    assert n_aps >= 1, (
        f"Expected at least 1 AP with 0.2 µA/cm² from v_rest=−65 mV, "
        f"but detected {n_aps}.  v_rest may no longer be the pacemaker threshold."
    )


# ---------------------------------------------------------------------------
# AP generation and complex spiking
# ---------------------------------------------------------------------------


def test_suprathreshold_fires_action_potential(pk_neuron: Neuron) -> None:
    """Suprathreshold pulse produces a full-amplitude action potential.

    Confirms that DSB94 Na⁺/K⁺ kinetics and the NaP/NaR additions still
    support normal AP generation.  Peak must exceed +20 mV and the cell
    must repolarize below −50 mV after the peak.
    """
    protocol = step_current(
        duration=100.0,
        current_amplitude=5.0,
        step_start=50.0,
        step_duration=5.0,
    )
    result = simulate_current_clamp(pk_neuron, current_external=protocol)
    voltage = result["voltage"]

    assert float(voltage.max()) > 20.0, (
        f"AP peak {voltage.max():.1f} mV is below expected +20 mV"
    )
    peak_idx = int(np.argmax(voltage))
    assert float(voltage[peak_idx:].min()) < -50.0, (
        f"Post-peak minimum {voltage[peak_idx:].min():.1f} mV — cell did not repolarize"
    )
    assert _count_action_potentials(voltage) >= 1


def test_complex_spiking_with_strong_stimulus(pk_neuron: Neuron) -> None:
    """Strong sustained current evokes complex Ca²⁺-driven spiking.

    10 µA/cm² for 180 ms should drive at least 3 action potentials through
    the ICaL/ICaT/IKCa interaction characteristic of Purkinje complex spikes.
    Ref: De Schutter & Bower (1994), J. Neurophysiol. 71:375.
    """
    protocol = step_current(
        duration=200.0,
        current_amplitude=10.0,
        step_start=0.0,
        step_duration=180.0,
    )
    result = simulate_current_clamp(pk_neuron, current_external=protocol)
    n_aps = _count_action_potentials(result["voltage"])
    assert n_aps >= 3, f"Expected at least 3 complex spikes with 10 µA/cm², got {n_aps}"
