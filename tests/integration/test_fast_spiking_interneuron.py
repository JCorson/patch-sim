"""Behavioral tests for the Fast-Spiking Interneuron preset.

Pins the fast-spiking phenotype: narrow APs, high firing rate, minimal
spike-frequency adaptation, deep AHP from Kv3.1 / delayed-rectifier K⁺.
Bands cite Erisir et al. (1999), Kawaguchi (1995), and Wang & Buzsáki
(1996).  Two metrics (peak voltage and AHP depth) currently fall outside
literature ranges and are marked ``xfail`` pending biology fixes.

Replaces the previous coverage gap: before this file the FS preset had no
suprathreshold-firing tests at all, only sag/stability checks.
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.neuron import Neuron
from patch_sim.presets import make_fast_spiking_interneuron
from patch_sim.protocols import step_current
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Standard suprathreshold step: 30 µA/cm² is ~1.2× the model's rheobase
# (~25 µA/cm² with this leaky preset; R_in ≈ 0.67 kΩ·cm² makes voltage
# excursions per µA much smaller than for high-R_in pyramidal cells).  The
# 500 ms named duration runs ~1250 ms of simulation under the default
# 100 kHz protocol sampling rate, more than enough for a stable mean shape.
_FS_STEP_DURATION_MS = 500.0
_FS_STEP_CURRENT = 30.0
_FS_REFERENCE = "Erisir et al. 1999 / Kawaguchi 1995 / Wang & Buzsáki 1996"


@pytest.fixture
def fs_neuron() -> Neuron:
    """Fast-spiking interneuron instance for all tests in this module."""
    return make_fast_spiking_interneuron()


@pytest.fixture
def fs_ap_shape_result(fs_neuron: Neuron):
    """AP analysis under the standard suprathreshold step.

    Used by every shape-band test in this module so each test does not
    re-run the simulation.
    """
    protocol = step_current(
        duration=_FS_STEP_DURATION_MS,
        current_amplitude=_FS_STEP_CURRENT,
    )
    result = simulate_current_clamp(fs_neuron, current_external=protocol)
    return analyze_aps_from_result(result)


def _ms_to_samples(ms: float) -> int:
    """Convert a duration in milliseconds to a simulation sample count."""
    return int(ms * SIM_SAMPLING_FREQ / 1000.0)


# ---------------------------------------------------------------------------
# Resting stability and basic firing
# ---------------------------------------------------------------------------


def test_fs_no_spontaneous_firing(fs_neuron: Neuron) -> None:
    """FS interneuron must not fire spontaneously with zero injected current.

    The leak balance was tuned so the cell rests near −65 mV; a non-zero
    spontaneous rate would indicate that g_NaL/g_KL or Pospischil channel
    steady-state currents have drifted.
    """
    protocol = np.zeros(_ms_to_samples(500) + 1)
    result = simulate_current_clamp(fs_neuron, current_external=protocol)
    ap = analyze_aps_from_result(result)
    assert ap.spike_count == 0, (
        f"Expected no spontaneous APs at rest, but detected {ap.spike_count}."
    )


def test_fs_suprathreshold_step_drives_high_frequency_firing(
    fs_ap_shape_result,
) -> None:
    """Suprathreshold step drives sustained high-frequency firing.

    The defining feature of fast-spiking cells (Erisir et al. 1999): firing
    rates remain >100 Hz throughout a sustained step, with no rate collapse
    from depolarization block.  ≥50 spikes establishes that the cell sustains
    repetitive firing rather than firing once and silencing.
    """
    assert_ap_shape(
        fs_ap_shape_result,
        reference=_FS_REFERENCE,
        firing_rate_hz=(100.0, 500.0),
        min_spike_count=50,
    )


# ---------------------------------------------------------------------------
# AP shape — fast-spiking phenotype, Erisir et al. (1999); Kawaguchi (1995).
# Each metric has its own test so future biology fixes can flip xfail
# markers one at a time.
# ---------------------------------------------------------------------------


def test_fs_ap_half_width_in_fs_range(fs_ap_shape_result) -> None:
    """Mean AP half-width matches the narrow fast-spiking phenotype (0.25–0.7 ms).

    Erisir et al. (1999) report half-widths of ~0.4 ms at 35 °C for cortical
    fast-spiking basket cells.  Lower bound widened to 0.25 ms because Kv3.1
    can produce APs as narrow as 0.2 ms at body temperature.
    """
    assert_ap_shape(
        fs_ap_shape_result,
        reference=_FS_REFERENCE,
        half_width_ms=(0.25, 0.7),
    )


def test_fs_ap_threshold_in_fs_range(fs_ap_shape_result) -> None:
    """Mean AP threshold falls in the FS-interneuron range (−55 to −40 mV)."""
    assert_ap_shape(
        fs_ap_shape_result,
        reference=_FS_REFERENCE,
        threshold_mv=(-55.0, -40.0),
    )


def test_fs_ap_peak_voltage_in_fs_range(fs_ap_shape_result) -> None:
    """Mean AP peak voltage falls within the FS range (+10 to +40 mV)."""
    assert_ap_shape(
        fs_ap_shape_result,
        reference=_FS_REFERENCE,
        peak_mv=(10.0, 40.0),
    )


def test_fs_ap_ahp_depth_in_fs_range(fs_ap_shape_result) -> None:
    """Mean AHP depth falls within the FS range (−78 to −60 mV)."""
    assert_ap_shape(
        fs_ap_shape_result,
        reference=_FS_REFERENCE,
        ahp_mv=(-78.0, -60.0),
    )


def test_fs_single_ap_repolarizes_cleanly(fs_neuron: Neuron) -> None:
    """A single evoked AP repolarizes within 2 ms; no post-spike plateau.

    Drives a single AP with the UI ACTION_POTENTIAL protocol (30 µA/cm² ×
    5 ms) and asserts that within 2 ms after the peak the voltage has
    dropped below −58 mV.  Cortical fast-spiking (PV⁺) interneuron APs are
    ~0.4 ms half-width with a deep, fast Kv3-driven AHP reaching −60 to
    −75 mV within ~1 ms of the peak (Erisir et al. 1999, J. Neurophysiol.
    82:2476; Wang & Buzsáki 1996, J. Neurosci. 16:6402); the >100 Hz
    sustained firing this preset supports requires each AP to clear
    threshold within ~1–2 ms.  A long plateau hanging at ~−30 mV after the
    spike would mean a window current is dominating repolarization — a
    pathology the mean half-width and AHP-depth metrics do not catch
    (half-width is measured at the spike midpoint, which the trace can
    cross cleanly even when V parks at −30 mV afterward; cf. SNc DA,
    PR #318).
    """
    pre_ms, stim_ms, post_ms = 10.0, 5.0, 50.0
    total_samples = _ms_to_samples(pre_ms + stim_ms + post_ms) + 1
    current = np.zeros(total_samples)
    pre_n = _ms_to_samples(pre_ms)
    stim_n = _ms_to_samples(stim_ms)
    current[pre_n : pre_n + stim_n] = 30.0
    result = simulate_current_clamp(fs_neuron, current_external=current)
    t = np.asarray(result["time"])
    v = np.asarray(result["voltage"])

    peak_idx = int(np.argmax(v))
    peak_t = float(t[peak_idx])
    after_mask = (t > peak_t) & (t <= peak_t + 2.0)
    v_after = v[after_mask]
    assert v_after.size > 0, "post-peak window is empty"
    min_v = float(np.min(v_after))
    assert min_v < -58.0, (
        f"AP plateau detected: V never falls below −58 mV in 2 ms after the "
        f"peak (min V = {min_v:.2f} mV at peak={peak_t:.2f} ms).  "
        f"Reference: {_FS_REFERENCE}."
    )


# ---------------------------------------------------------------------------
# Firing pattern — non-adapting (Erisir et al. 1999).
# ---------------------------------------------------------------------------


def test_fs_low_spike_frequency_adaptation(fs_ap_shape_result) -> None:
    """Inter-spike intervals stay roughly constant — non-adapting FS phenotype.

    The defining characteristic of fast-spiking cells (Erisir et al. 1999):
    repetitive firing without significant frequency adaptation, supported by
    Kv3.1's fast deactivation.  We assert that the mean of the last 10 ISIs
    is no more than 30 % larger than the mean of the first 10, comfortably
    above noise but well below the 2–5× ratio seen in adapting RS cells.
    """
    isis = fs_ap_shape_result.isis
    assert len(isis) >= 30, (
        f"Need ≥30 ISIs to compare early vs late firing, got {len(isis)}"
    )
    mean_early = float(np.mean(isis[:10]))
    mean_late = float(np.mean(isis[-10:]))
    ratio = mean_late / mean_early
    assert ratio < 1.3, (
        f"FS interneurons should be non-adapting (mean late / early ISI < 1.3); "
        f"got mean_early={mean_early:.2f} ms, mean_late={mean_late:.2f} ms, "
        f"ratio={ratio:.2f}.  Reference: Erisir et al. (1999)."
    )


def test_fs_nav11_slow_inactivation_does_not_dominate_firing(
    fs_neuron: Neuron,
) -> None:
    """The Nav1.1 sNa11 slow gate must barely engage at FSI firing rates.

    The whole point of giving FSI a *weak* Nav1.1-flavoured slow gate
    (V½ = −45 mV, τ_floor = 5000 ms — see make_nav11_channel; Patel et
    al. 2015 PLOS ONE 10:e0133485 for the underlying biology) instead of
    no slow inactivation at all is that the gate captures the biological
    fact that Nav1.1 slow-inactivates while not suppressing the
    100–500 Hz fast-spiking phenotype (Hu & Jonas 2014, Nat. Neurosci.
    17:686).

    This test asserts the gate barely moves over a 500 ms suprathreshold
    step: sNa11 at the end of the step must remain ≥ 0.85 of its rest
    availability.  If it falls below that, FSI biology is being
    suppressed by the gate kinetics and the τ_floor needs further
    slowing or the FSI g_Na needs an upward retune.
    """
    n_pre = _ms_to_samples(50.0)
    n_step = _ms_to_samples(500.0)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, _FS_STEP_CURRENT),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(fs_neuron, current_external=current)
    assert result.dtype.names is not None and "sNa11" in result.dtype.names, (
        f"FSI Na channel must expose sNa11 column (got {result.dtype.names})"
    )
    sNa11_at_rest = float(result["sNa11"][n_pre - 1])
    sNa11_at_step_end = float(result["sNa11"][n_pre + n_step - 1])
    fraction_remaining = sNa11_at_step_end / sNa11_at_rest
    assert fraction_remaining >= 0.85, (
        f"Nav1.1 slow gate accumulated too much inactivation at FSI firing "
        f"rates: rest={sNa11_at_rest:.3f}, step end={sNa11_at_step_end:.3f}, "
        f"fraction remaining={fraction_remaining:.3f} (expected ≥ 0.85). "
        "Either τ_scale on _nav11_alpha_sNa needs further slowing, or g_Na "
        "needs an upward retune to compensate."
    )
