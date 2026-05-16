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
from patch_sim.clamp_simulations import simulate_current_clamp
from patch_sim.constants import SIM_SAMPLING_FREQ
from patch_sim.neuron import Neuron
from patch_sim.presets import make_squid_giant_axon
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
    return make_squid_giant_axon()


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


# ---------------------------------------------------------------------------
# Post-spike repolarization — fall-time guard (#321).
# ---------------------------------------------------------------------------


def test_squid_single_ap_repolarizes_cleanly(squid_neuron: Neuron) -> None:
    """A single evoked AP repolarizes within 5 ms; no post-spike plateau.

    Drives a single AP with the UI ACTION_POTENTIAL protocol (10 µA/cm² ×
    10 ms) and asserts that within 5 ms after the peak the voltage has
    dropped below −60 mV.  The HH52 squid AP is ~1 ms wide and the fast
    K⁺-driven undershoot reaches near E_K (≈ −72 mV in the original model)
    within a few ms of the peak (Hodgkin & Huxley 1952, J. Physiol.
    117:500); this is also implied by the 40–120 Hz repetitive firing the
    preset sustains, which requires each AP to clear well below threshold
    within ~5 ms.  A long plateau hanging at ~−30 mV after the spike would
    mean a window current is dominating repolarization — a pathology the
    mean half-width and AHP-depth metrics do not catch (half-width is
    measured at the spike midpoint, which the trace can cross cleanly even
    when V parks at −30 mV afterward; cf. SNc DA, PR #318).
    """
    pre_ms, stim_ms, post_ms = 10.0, 10.0, 50.0
    total_samples = _ms_to_samples(pre_ms + stim_ms + post_ms) + 1
    current = np.zeros(total_samples)
    pre_n = _ms_to_samples(pre_ms)
    stim_n = _ms_to_samples(stim_ms)
    current[pre_n : pre_n + stim_n] = 10.0
    result = simulate_current_clamp(squid_neuron, current_external=current)
    t = np.asarray(result["time"])
    v = np.asarray(result["voltage"])

    peak_idx = int(np.argmax(v))
    peak_t = float(t[peak_idx])
    after_mask = (t > peak_t) & (t <= peak_t + 5.0)
    v_after = v[after_mask]
    assert v_after.size > 0, "post-peak window is empty"
    min_v = float(np.min(v_after))
    assert min_v < -60.0, (
        f"AP plateau detected: V never falls below −60 mV in 5 ms after the "
        f"peak (min V = {min_v:.2f} mV at peak={peak_t:.2f} ms).  "
        f"Reference: {_HH52_REFERENCE}."
    )
