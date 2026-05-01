"""Behavioral tests for the Subthalamic Nucleus (STN) neuron preset.

Pins the STN tonic-pacemaker phenotype: 5–50 Hz autonomous firing, AP
shape consistent with Bevan & Wilson (1999), and the conditional burst
mode (already covered by ``test_stn_conditional_burst_mode_under_…`` in
``test_burst_metrics_simulation.py``).  Bands cite Beurrier et al.
(1999) and Bevan & Wilson (1999).  Several metrics — most notably the
absence of spontaneous pacemaking and an excessive tonic rate under the
documented bias — currently fall outside literature ranges and are
marked ``xfail`` pending biology fixes.

Replaces the previous coverage gap: before this file the STN preset had
only sag and conditional-burst tests.
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import STN
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Tonic bias documented in the preset comment ("~80 Hz here under a 2 µA/cm²
# depolarising bias").  Use the documented stim so that future biology-fix
# work can compare against the same reference operating point.
_STN_STEP_DURATION_MS = 200.0
_STN_STEP_CURRENT = 2.0
_STN_REFERENCE = "Beurrier et al. 1999 / Bevan & Wilson 1999"


@pytest.fixture
def stn_neuron() -> Neuron:
    """STN neuron instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[STN])


@pytest.fixture
def stn_ap_shape_result(stn_neuron: Neuron):
    """AP analysis under the documented 2 µA/cm² tonic bias."""
    protocol = step_current(
        duration=_STN_STEP_DURATION_MS,
        current_amplitude=_STN_STEP_CURRENT,
    )
    result = simulate_current_clamp(stn_neuron, current_external=protocol)
    return analyze_aps_from_result(result)


def _ms_to_samples(ms: float) -> int:
    """Convert ms to simulation samples at SIM_SAMPLING_FREQ."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Spontaneous tonic pacemaking — Bevan & Wilson (1999), 5–50 Hz in vitro.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "STN preset is labelled an autonomous tonic pacemaker but produces "
        "ZERO spikes in 1 s of zero-current simulation (preset documents "
        "needing a 2 µA/cm² bias to fire). Bevan & Wilson 1999 report 5–50 "
        "Hz autonomous firing in slice without external bias. Likely the "
        "Otsuka kinetics' steady-state currents at v_rest = −68 mV do not "
        "drive spontaneous depolarisation; tracked in #305."
    ),
)
def test_stn_spontaneous_pacemaking(stn_neuron: Neuron) -> None:
    """STN fires autonomously at 5–50 Hz without injected current.

    Bevan & Wilson (1999) characterise the STN as a tonic pacemaker
    sustained intrinsically by Na⁺/K⁺ window currents and Ih.
    """
    duration_ms = 1000.0
    zero_current = np.zeros(_ms_to_samples(duration_ms) + 1)
    result = simulate_current_clamp(stn_neuron, current_external=zero_current)
    ap = analyze_aps_from_result(result)
    assert_ap_shape(
        ap,
        reference=_STN_REFERENCE,
        firing_rate_hz=(5.0, 50.0),
        min_spike_count=5,
    )


# ---------------------------------------------------------------------------
# Tonic mode under the documented depolarising bias.
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "STN fires at ~134 Hz under the documented 2 µA/cm² bias, well "
        "above the 5–50 Hz tonic range reported by Bevan & Wilson (1999). "
        "Either g_Na is too high or the leak balance pushes the cell into "
        "an over-excited regime; tracked in #305."
    ),
)
def test_stn_tonic_firing_rate_in_stn_range(stn_ap_shape_result) -> None:
    """Tonic firing rate falls within the STN range (5–50 Hz)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        firing_rate_hz=(5.0, 50.0),
        min_spike_count=2,
    )


# ---------------------------------------------------------------------------
# AP shape — STN tonic phenotype, Bevan & Wilson (1999).
# ---------------------------------------------------------------------------


def test_stn_ap_threshold_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP threshold falls in the STN range (−60 to −40 mV)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        threshold_mv=(-60.0, -40.0),
    )


def test_stn_ap_peak_voltage_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP peak voltage falls within the STN range (0 to +30 mV).

    STN APs have a comparatively low overshoot relative to other neurons —
    Bevan & Wilson (1999) report typical AP amplitudes of 60–80 mV from
    a threshold near −55 mV, putting the peak in the +5 to +25 mV range.
    """
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        peak_mv=(0.0, 30.0),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Mean AP half-width ~0.12 ms is far below the 0.4–1.2 ms range "
        "reported for STN tonic spikes (Bevan & Wilson 1999). The Otsuka "
        "kinetics under this preset's parameters produce excessively narrow "
        "APs; tracked in #305."
    ),
)
def test_stn_ap_half_width_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AP half-width matches the STN tonic phenotype (0.4–1.2 ms)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        half_width_ms=(0.4, 1.2),
    )


def test_stn_ap_ahp_depth_in_stn_range(stn_ap_shape_result) -> None:
    """Mean AHP depth falls within the STN range (−85 to −55 mV)."""
    assert_ap_shape(
        stn_ap_shape_result,
        reference=_STN_REFERENCE,
        ahp_mv=(-85.0, -55.0),
    )
