"""AP-shape tests for the Squid Giant Axon (HH52) preset.

Pins the classic Hodgkin-Huxley AP shape: ~1 ms half-width, +30 to +50 mV
peak, fast-AHP near E_K (~−75 mV), threshold around −50 mV.  This is the
reference preparation, so the bands are tightest here.

Existing squid tests live across multiple integration files
(``test_ap_metrics_simulation.py``, ``test_fi_curve_simulation.py``,
``test_hyperpolarization_simulation.py``, etc.).  This module is the
single place where the *AP shape* is asserted against literature; the
others continue to exercise rheobase, F-I, sag, and so on.
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import SQUID_GIANT_AXON
from patch_sim.neuron import Neuron
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Standard suprathreshold step: 10 µA/cm² is a textbook value for HH52
# repetitive firing in slice / model output.  100 ms gives ~10 APs at the
# squid's typical 60–100 Hz repetitive rate, enough for a stable shape mean.
_HH52_STEP_DURATION_MS = 100.0
_HH52_STEP_CURRENT = 10.0
_HH52_REFERENCE = "Hodgkin & Huxley 1952"


@pytest.fixture
def squid_neuron() -> Neuron:
    """HH52 squid giant axon instance for all tests in this module."""
    return make_neuron(NEURON_PRESETS[SQUID_GIANT_AXON])


@pytest.fixture
def squid_ap_shape_result(squid_neuron: Neuron):
    """AP analysis under the standard HH52 suprathreshold step."""
    protocol = step_current(
        duration=_HH52_STEP_DURATION_MS,
        current_amplitude=_HH52_STEP_CURRENT,
    )
    result = simulate_current_clamp(squid_neuron, current_external=protocol)
    return analyze_aps_from_result(result)


def _ms_to_samples(ms: float) -> int:
    """Convert ms to simulation samples at SIM_SAMPLING_FREQ."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Resting state — squid axon is silent at rest with the HH52 leak balance.
# ---------------------------------------------------------------------------


def test_squid_no_spontaneous_firing(squid_neuron: Neuron) -> None:
    """Squid axon must not fire spontaneously at rest.

    HH52 has no pacemaker mechanisms; with zero injected current the axon
    sits at rest near −65 mV.
    """
    zero_current = np.zeros(_ms_to_samples(200) + 1)
    result = simulate_current_clamp(squid_neuron, current_external=zero_current)
    ap = analyze_aps_from_result(result)
    assert ap.spike_count == 0, (
        f"Expected no spontaneous APs, but detected {ap.spike_count}."
    )


# ---------------------------------------------------------------------------
# AP shape — Hodgkin & Huxley (1952) classic squid axon values.  No xfail
# markers expected: HH52 is the reference preparation and the model
# faithfully reproduces these values.
# ---------------------------------------------------------------------------


def test_squid_ap_half_width_in_hh52_range(squid_ap_shape_result) -> None:
    """Mean AP half-width matches HH52 squid axon (0.6–2.0 ms).

    The classic HH52 AP at 6.3 °C has a half-width of approximately 1 ms;
    this preset uses Q10=1.0 (room-temperature-equivalent) so the model
    output should land in the same range.
    """
    assert_ap_shape(
        squid_ap_shape_result,
        reference=_HH52_REFERENCE,
        half_width_ms=(0.6, 2.0),
    )


def test_squid_ap_peak_voltage_in_hh52_range(squid_ap_shape_result) -> None:
    """Mean AP peak voltage matches HH52 (+20 to +50 mV)."""
    assert_ap_shape(
        squid_ap_shape_result,
        reference=_HH52_REFERENCE,
        peak_mv=(20.0, 50.0),
    )


def test_squid_ap_threshold_in_hh52_range(squid_ap_shape_result) -> None:
    """Mean AP threshold matches HH52 (−55 to −40 mV)."""
    assert_ap_shape(
        squid_ap_shape_result,
        reference=_HH52_REFERENCE,
        threshold_mv=(-55.0, -40.0),
    )


def test_squid_ap_ahp_depth_in_hh52_range(squid_ap_shape_result) -> None:
    """Mean AHP depth matches HH52 fast-AHP near E_K (−85 to −65 mV)."""
    assert_ap_shape(
        squid_ap_shape_result,
        reference=_HH52_REFERENCE,
        ahp_mv=(-85.0, -65.0),
    )


def test_squid_repetitive_firing_rate_in_hh52_range(squid_ap_shape_result) -> None:
    """Repetitive firing rate at 10 µA/cm² falls in the HH52 range (40–120 Hz)."""
    assert_ap_shape(
        squid_ap_shape_result,
        reference=_HH52_REFERENCE,
        firing_rate_hz=(40.0, 120.0),
        min_spike_count=5,
    )
