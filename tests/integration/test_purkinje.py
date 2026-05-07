"""Behavioral tests for the Purkinje neuron preset.

Verifies spontaneous pacemaking driven by Ih (hyperpolarization-activated
current) and the NaP/NaR window-current complex: the cell fires autonomously
without external current, INaP provides persistent inward current near
threshold, and INaR is tracked in the simulation output.

Also pins the pacemaker AP shape (half-width, peak, threshold, AHP, rate)
against literature-cited tolerance bands from Häusser & Clark (1997) and
Raman & Bean (1999).
"""

import numpy as np
import pytest

from patch_sim.analysis.ap_metrics import analyze_aps_from_result
from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import PURKINJE
from patch_sim.neuron import Neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current
from tests.integration._ap_shape import assert_ap_shape

# ---------------------------------------------------------------------------
# Shared fixture and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def pk_neuron() -> Neuron:
    """Purkinje neuron instance for all tests in this module."""
    return NEURON_PRESETS[PURKINJE]()


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
        elif v <= threshold:
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
# Spontaneous pacemaking
# ---------------------------------------------------------------------------


def test_fires_spontaneously(pk_neuron: Neuron) -> None:
    """Purkinje cell fires spontaneously with zero external current.

    With Ih (g=1.0 mS/cm²) recovering the cell from the post-AP AHP and INaP
    destabilising the rest near −65 mV, the cell should sustain autonomous
    pacemaking.  At least 5 APs must occur in 500 ms of zero current,
    confirming a mean inter-spike interval ≤ 100 ms (≥ 10 Hz).
    Ref: Häusser & Clark (1997), J. Neurosci. 17:2358.
    """
    zero_current = np.zeros(_ms_to_samples(500) + 1)
    result = simulate_current_clamp(pk_neuron, current_external=zero_current)
    n_aps = _count_action_potentials(result["voltage"])
    assert n_aps >= 5, (
        f"Expected ≥5 spontaneous APs in 500 ms (≥10 Hz pacemaking), "
        f"but detected {n_aps}.  Ih may not be wired into the preset or "
        "g_Ih may be too small to drive recovery from the post-AP AHP."
    )


# ---------------------------------------------------------------------------
# INaR: resurgent Na⁺ channel presence
# ---------------------------------------------------------------------------


def test_inar_column_present_in_simulation(pk_neuron: Neuron) -> None:
    """INaR current column appears in the simulation result dtype.

    Confirms that the INaR channel is wired into the preset and that the
    resurgent sodium current is tracked in the output structured array.
    Ref: Raman & Bean (1997), Neuron 19:881.
    """
    protocol = step_current(
        duration=50.0,
        current_amplitude=5.0,
        sampling_frequency=SIM_SAMPLING_FREQ,
    )
    result = simulate_current_clamp(pk_neuron, current_external=protocol)
    assert "INaR" in result.dtype.names, (
        "Expected 'INaR' column in simulation output — make_inar_channel may "
        "not be included in the Purkinje preset channels tuple"
    )


# ---------------------------------------------------------------------------
# Ih: hyperpolarization-activated current presence
# ---------------------------------------------------------------------------


def test_ih_column_present_in_simulation(pk_neuron: Neuron) -> None:
    """Ih current column appears in the simulation result dtype.

    Confirms that the Ih channel is wired into the preset and that the
    hyperpolarization-activated cation current is tracked in the output.
    Ref: Destexhe et al. (1993), J. Neurophysiol. 70:1385.
    """
    zero_current = np.zeros(_ms_to_samples(100) + 1)
    result = simulate_current_clamp(pk_neuron, current_external=zero_current)
    assert "Ih" in result.dtype.names, (
        "Expected 'Ih' column in simulation output — make_ih_channel may "
        "not be included in the Purkinje preset channels tuple"
    )


# ---------------------------------------------------------------------------
# AP generation and complex spiking
# ---------------------------------------------------------------------------


def test_suprathreshold_fires_action_potential(pk_neuron: Neuron) -> None:
    """Suprathreshold pulse produces a full-amplitude action potential.

    Confirms that DSB94 Na⁺/K⁺ kinetics and the NaP/NaR/Ih additions still
    support normal AP generation.  Peak must exceed +20 mV and the cell
    must repolarize below −50 mV after the peak.
    """
    protocol = step_current(
        duration=50.0,
        current_amplitude=5.0,
        step_start=10.0,
        step_duration=5.0,
        sampling_frequency=SIM_SAMPLING_FREQ,
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
    """Strong sustained current evokes multiple action potentials.

    10 µA/cm² for 180 ms should drive at least 3 action potentials through
    the Ca²⁺/IKCa interaction characteristic of Purkinje sustained firing.
    Ref: De Schutter & Bower (1994), J. Neurophysiol. 71:375.
    """
    protocol = step_current(
        duration=200.0,
        current_amplitude=10.0,
        step_start=0.0,
        step_duration=180.0,
        sampling_frequency=SIM_SAMPLING_FREQ,
    )
    result = simulate_current_clamp(pk_neuron, current_external=protocol)
    n_aps = _count_action_potentials(result["voltage"])
    assert n_aps >= 6, f"Expected at least 6 complex spikes with 10 µA/cm², got {n_aps}"


# ---------------------------------------------------------------------------
# AP shape — pacemaker phenotype, Häusser & Clark (1997);
# Raman & Bean (1999).  Each metric has its own test so future biology fixes
# can flip xfail markers one at a time.  Shape is measured under spontaneous
# (zero-current) pacemaking, the canonical Purkinje protocol.
# ---------------------------------------------------------------------------

_PK_REFERENCE = "Häusser & Clark 1997 / Raman & Bean 1999"
_PK_SPONTANEOUS_DURATION_MS = 500.0


@pytest.fixture
def pk_pacemaker_ap_result(pk_neuron: Neuron):
    """AP analysis of Purkinje under zero-current spontaneous pacemaking.

    Mirrors the protocol used by ``test_fires_spontaneously`` so the shape
    bands and the qualitative pacemaking assertion live on the same trace.
    """
    zero_current = np.zeros(_ms_to_samples(_PK_SPONTANEOUS_DURATION_MS) + 1)
    result = simulate_current_clamp(pk_neuron, current_external=zero_current)
    return analyze_aps_from_result(result)


def test_pk_spontaneous_firing_rate_in_pacemaker_range(pk_pacemaker_ap_result) -> None:
    """Spontaneous firing rate falls within the pacemaker range (10–50 Hz).

    Häusser & Clark (1997) report rat Purkinje cells firing autonomously
    around 30–90 Hz at 32–37 °C in slice; rates drop closer to 10–30 Hz at
    room temperature.  Tolerance band 10–50 Hz covers both regimes.
    """
    assert_ap_shape(
        pk_pacemaker_ap_result,
        reference=_PK_REFERENCE,
        firing_rate_hz=(10.0, 50.0),
        min_spike_count=5,
    )


def test_pk_ap_half_width_in_pacemaker_range(pk_pacemaker_ap_result) -> None:
    """Mean AP half-width matches the narrow Purkinje phenotype (0.2–0.6 ms).

    Raman & Bean (1999) report half-widths in the 0.2–0.5 ms range for
    Purkinje cells in slice; 0.6 ms upper bound allows for room-temperature
    broadening.
    """
    assert_ap_shape(
        pk_pacemaker_ap_result,
        reference=_PK_REFERENCE,
        half_width_ms=(0.2, 0.6),
    )


def test_pk_ap_threshold_in_pacemaker_range(pk_pacemaker_ap_result) -> None:
    """Mean AP threshold falls in the literature Purkinje range (−55 to −40 mV)."""
    assert_ap_shape(
        pk_pacemaker_ap_result,
        reference=_PK_REFERENCE,
        threshold_mv=(-55.0, -40.0),
    )


def test_pk_ap_peak_voltage_in_pacemaker_range(pk_pacemaker_ap_result) -> None:
    """Mean AP peak voltage falls within the Purkinje range (+10 to +40 mV)."""
    assert_ap_shape(
        pk_pacemaker_ap_result,
        reference=_PK_REFERENCE,
        peak_mv=(10.0, 40.0),
    )


def test_pk_ap_ahp_depth_in_pacemaker_range(pk_pacemaker_ap_result) -> None:
    """Mean AHP depth falls within the Purkinje range (−55 to −72 mV)."""
    assert_ap_shape(
        pk_pacemaker_ap_result,
        reference=_PK_REFERENCE,
        ahp_mv=(-72.0, -55.0),
    )


# ---------------------------------------------------------------------------
# Slow Na inactivation engagement and depol-block recovery — issue #329.
# ---------------------------------------------------------------------------

_PK_DEPOL_DRIVE_AMP_UA = 10.0
_PK_DEPOL_DRIVE_MS = 200.0


def test_pk_inap_slow_inactivation_engages_during_drive(pk_neuron: Neuron) -> None:
    """The sNaP gate closes during +10 µA/cm² × 200 ms drive (#329 mechanism).

    Direct mechanism check for #329: the Magistretti & Alonso 1999 slow
    inactivation gate baked into ``make_inap_channel`` must be doing
    real work during sustained suprathreshold drive.  By the end of the step,
    sNaP should have lost at least half its rest availability so the
    persistent Na⁺ contribution to the depol-block plateau is suppressed.
    The +10 µA/cm² level matches the existing
    ``test_complex_spiking_with_strong_stimulus`` step current and the
    issue body's specified amplitude.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(_PK_DEPOL_DRIVE_MS)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, _PK_DEPOL_DRIVE_AMP_UA),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(pk_neuron, current_external=current)
    sNaP_at_rest = float(result["sNaP"][n_pre - 1])
    assert sNaP_at_rest > 0.5, (
        f"sNaP rest availability is unexpectedly low ({sNaP_at_rest:.3f}); "
        "the inactivation engagement check below is meaningless if the "
        "gate is already mostly closed at v_rest."
    )
    sNaP_at_step_end = float(result["sNaP"][n_pre + n_step - 1])
    fraction_inactivated = 1.0 - sNaP_at_step_end / sNaP_at_rest
    assert fraction_inactivated > 0.5, (
        f"sNaP did not meaningfully inactivate: rest={sNaP_at_rest:.3f}, "
        f"step end={sNaP_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.5)"
    )


def test_pk_fast_na_slow_inactivation_engages_during_drive(
    pk_neuron: Neuron,
) -> None:
    """The sNa gate (fast Na slow inactivation) closes during +10 µA/cm² × 200 ms.

    Direct mechanism check for #329: the Carter & Bean 2009 slow voltage-
    dependent inactivation gate baked into ``make_purkinje_na_channel``
    must lose at least half its rest availability by the end of a sustained
    suprathreshold step, abolishing the residual fast-Na h-tail that would
    otherwise pin the cell on a depolarisation plateau.  Carter & Bean 2009
    directly studied cerebellar Purkinje cells, so this is the cell type
    the paper actually characterised.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(_PK_DEPOL_DRIVE_MS)
    n_post = _ms_to_samples(50.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, _PK_DEPOL_DRIVE_AMP_UA),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(pk_neuron, current_external=current)
    sNa_at_rest = float(result["sNa"][n_pre - 1])
    assert sNa_at_rest > 0.5, (
        f"sNa rest availability is unexpectedly low ({sNa_at_rest:.3f}); "
        "the inactivation engagement check below is meaningless if the "
        "gate is already mostly closed at v_rest."
    )
    sNa_at_step_end = float(result["sNa"][n_pre + n_step - 1])
    fraction_inactivated = 1.0 - sNa_at_step_end / sNa_at_rest
    assert fraction_inactivated > 0.5, (
        f"sNa did not meaningfully inactivate: rest={sNa_at_rest:.3f}, "
        f"step end={sNa_at_step_end:.3f}, "
        f"fraction inactivated={fraction_inactivated:.2f} (expected > 0.5)"
    )


def test_pk_recovers_from_sustained_suprathreshold_drive(
    pk_neuron: Neuron,
) -> None:
    """Purkinje repolarises after +10 µA/cm² × 200 ms (#329 regression).

    Without slow Na inactivation the cell can hang on a depol-block
    plateau under sustained drive (cf. STN #324, cortical pyramidal
    #327, CA1 pyramidal #328).  With sNaP (Magistretti & Alonso 1999)
    and Purkinje sNa (Carter & Bean 2009) both present, the cell must
    escape the plateau within the post-stim window and settle below
    −50 mV during the last 200 ms of a 700 ms post-stimulus epoch.
    """
    n_pre = _ms_to_samples(100.0)
    n_step = _ms_to_samples(_PK_DEPOL_DRIVE_MS)
    n_post = _ms_to_samples(700.0)
    current = np.concatenate(
        [
            np.zeros(n_pre),
            np.full(n_step, _PK_DEPOL_DRIVE_AMP_UA),
            np.zeros(n_post + 1),
        ]
    )
    result = simulate_current_clamp(pk_neuron, current_external=current)
    tail = np.asarray(result["voltage"][-_ms_to_samples(200.0) :])
    mean_v = float(tail.mean())
    assert mean_v < -50.0, (
        f"Purkinje failed to recover from +10 µA/cm² × 200 ms; "
        f"mean V in last 200 ms = {mean_v:.2f} mV"
    )


def test_pk_sNa_and_sNaP_columns_present(pk_neuron: Neuron) -> None:
    """``sNa`` and ``sNaP`` columns appear in the simulation output.

    Direct check of the issue #329 acceptance criterion that both slow
    inactivation gating variables surface as named columns.  The Purkinje
    preset is unique in that it concurrently runs INaR (with its own ``s``
    activation gate), INaP (with its ``sNaP`` slow inactivation gate),
    and the fast-Na ``sNa`` slow inactivation gate; this test pins the
    namespacing so the three gate names cannot collide.
    """
    n_samples = _ms_to_samples(50.0) + 1
    zero_current = np.zeros(n_samples)
    result = simulate_current_clamp(pk_neuron, current_external=zero_current)
    names = result.dtype.names or ()
    assert "sNa" in names, (
        f"Expected 'sNa' column for fast-Na slow inactivation; got {names!r}"
    )
    assert "sNaP" in names, (
        f"Expected 'sNaP' column for INaP slow inactivation; got {names!r}"
    )
    assert "s" in names, (
        f"Expected 'INaR' activation 's' column; got {names!r}.  Loss of "
        "this column would indicate a regression in INaR wiring."
    )
