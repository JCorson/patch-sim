"""Behavioral tests for the Cortical Pyramidal neuron preset.

Verifies that the three auxiliary channel behaviors — voltage sag (Ih),
subthreshold amplification (INaP), and spike-frequency adaptation (IM) —
are present after the conductance reductions in PR #206, and that the
reduced conductances prevent spontaneous tonic firing.
"""

import numpy as np
import pytest

from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import CORTICAL_PYRAMIDAL
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current

# ---------------------------------------------------------------------------
# Shared fixture and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cp_neuron() -> Neuron:
    """Cortical Pyramidal neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[CORTICAL_PYRAMIDAL])


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
# Resting stability
# ---------------------------------------------------------------------------


def test_no_spontaneous_firing(cp_neuron: Neuron) -> None:
    """Neuron must not fire spontaneously with zero injected current.

    Regression test for the conductance reductions in PR #206: the original
    g_h=1.5 and g_NaP=0.5 values caused excessive combined inward current
    that produced tonic firing even without external stimulation.  The
    reduced values (g_h=0.3, g_NaP=0.1) should eliminate this.
    """
    protocol = np.zeros(_ms_to_samples(200) + 1)
    result = simulate_current_clamp(cp_neuron, current_external=protocol)

    n_aps = _count_action_potentials(result["voltage"])
    assert n_aps == 0, (
        f"Expected no spontaneous APs, but detected {n_aps} "
        "(g_h or g_NaP may be too large)"
    )


# ---------------------------------------------------------------------------
# Ih: voltage sag on hyperpolarization
# ---------------------------------------------------------------------------


def test_voltage_sag_on_hyperpolarization(cp_neuron: Neuron) -> None:
    """Hyperpolarizing step current produces a depolarizing voltage sag.

    Ih is a hyperpolarization-activated inward cation current.  When a
    sustained negative current is injected, the membrane first hyperpolarizes
    to a peak, then Ih activates and partially depolarizes the cell back
    toward rest — the characteristic 'sag'.  Asserting that the steady-state
    voltage is more positive than the initial peak confirms Ih is active.
    """
    duration_ms = 300.0
    protocol = step_current(
        duration=duration_ms,
        current_amplitude=-5.0,
        step_start=50.0,
    )
    result = simulate_current_clamp(cp_neuron, current_external=protocol)
    voltage = result["voltage"]

    # Find the peak hyperpolarization during the step (after step onset).
    step_start_sample = _ms_to_samples(50)
    step_end_sample = _ms_to_samples(duration_ms)
    step_voltage = voltage[step_start_sample:step_end_sample]

    peak_hyperpolarization = float(step_voltage.min())

    # Steady-state: mean of the last 50 ms of the step.
    steady_state_start = _ms_to_samples(250)
    steady_state_voltage = float(voltage[steady_state_start:step_end_sample].mean())

    sag_amplitude = steady_state_voltage - peak_hyperpolarization
    assert sag_amplitude > 1.0, (
        f"Expected voltage sag > 1 mV from Ih activation, "
        f"got {sag_amplitude:.2f} mV "
        f"(peak={peak_hyperpolarization:.1f} mV, "
        f"steady-state={steady_state_voltage:.1f} mV)"
    )


# ---------------------------------------------------------------------------
# INaP: subthreshold amplification
# ---------------------------------------------------------------------------


def test_subthreshold_amplification(cp_neuron: Neuron) -> None:
    """INaP carries inward current during subthreshold depolarizations.

    The persistent sodium current activates near resting potential and
    provides a depolarizing (inward, negative) current that amplifies small
    voltage excursions.  With a subthreshold depolarizing step, the INaP
    column in the simulation result should be negative throughout the step,
    confirming the channel is active and contributing to amplification.
    """
    protocol = step_current(
        duration=200.0,
        current_amplitude=1.0,  # µA/cm² — well below AP threshold
        step_start=50.0,
        step_duration=100.0,
    )
    result = simulate_current_clamp(cp_neuron, current_external=protocol)

    # Confirm the step remains subthreshold.
    step_start_sample = _ms_to_samples(50)
    step_end_sample = _ms_to_samples(150)
    peak_voltage = float(result["voltage"][step_start_sample:step_end_sample].max())
    assert peak_voltage < 0.0, (
        f"Step should be subthreshold, but voltage reached {peak_voltage:.1f} mV"
    )

    # INaP is an inward Na⁺ current: negative values mean depolarizing current.
    inap_during_step = result["INaP"][step_start_sample:step_end_sample]
    assert float(inap_during_step.max()) < 0.0, (
        f"Expected INaP to be inward (negative) during depolarization, "
        f"but max was {inap_during_step.max():.4f} µA/cm²"
    )
    # INaP magnitude should be physiologically non-trivial (> 0.1 µA/cm²).
    assert float(inap_during_step.min()) < -0.1, (
        f"INaP magnitude {abs(inap_during_step.min()):.4f} µA/cm² is too small "
        "to meaningfully amplify subthreshold inputs"
    )


# ---------------------------------------------------------------------------
# IM: spike-frequency adaptation
# ---------------------------------------------------------------------------


def test_spike_frequency_adaptation(cp_neuron: Neuron) -> None:
    """Sustained suprathreshold current produces increasing inter-spike intervals.

    IM is a slow, non-inactivating K⁺ current that accumulates during
    repetitive firing and progressively reduces excitability.  With a long
    suprathreshold step, the inter-spike intervals (ISIs) should grow over
    time as IM builds up — the mean of the last three ISIs must exceed the
    mean of the first three.
    """
    protocol = step_current(duration=800.0, current_amplitude=5.0)
    result = simulate_current_clamp(cp_neuron, current_external=protocol)
    voltage = result["voltage"]
    time = result["time"]

    # Collect spike times as upward zero-crossings.
    spike_times: list[float] = []
    above = False
    for t, v in zip(time, voltage):
        if v > 0.0 and not above:
            spike_times.append(float(t))
            above = True
        elif v < 0.0:
            above = False

    assert len(spike_times) >= 6, (
        f"Expected at least 6 spikes to compare ISIs, got {len(spike_times)}"
    )

    isis = np.diff(spike_times)
    mean_early_isi = float(isis[:3].mean())
    mean_late_isi = float(isis[-3:].mean())

    assert mean_late_isi > mean_early_isi, (
        f"Expected spike-frequency adaptation (late ISI > early ISI), "
        f"got early mean ISI={mean_early_isi:.2f} ms, "
        f"late mean ISI={mean_late_isi:.2f} ms"
    )


# ---------------------------------------------------------------------------
# Basic firing
# ---------------------------------------------------------------------------


def test_suprathreshold_fires_action_potentials(cp_neuron: Neuron) -> None:
    """Suprathreshold step current produces full-amplitude action potentials.

    Confirms that the Pospischil Na⁺/K⁺ kinetics and reduced auxiliary
    channel conductances still support normal action potential generation.
    """
    protocol = step_current(duration=100.0, current_amplitude=10.0)
    result = simulate_current_clamp(cp_neuron, current_external=protocol)
    voltage = result["voltage"]

    # AP peak must exceed +20 mV.
    assert float(voltage.max()) > 20.0, (
        f"AP peak {voltage.max():.1f} mV is below expected +20 mV"
    )

    # Repolarisation must return below -50 mV after the peak.
    peak_idx = int(np.argmax(voltage))
    post_peak = voltage[peak_idx:]
    assert float(post_peak.min()) < -50.0, (
        f"Post-peak minimum {post_peak.min():.1f} mV — neuron did not repolarize"
    )

    # At least one AP must be detected.
    n_aps = _count_action_potentials(voltage)
    assert n_aps >= 1, f"Expected at least 1 AP, detected {n_aps}"
