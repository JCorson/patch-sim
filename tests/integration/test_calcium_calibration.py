"""Integration tests that assert peak ca_i lands in the physiological band.

Each Ca²⁺-carrying preset is run under its production strong-stimulus protocol
(matching NEURON_PROTOCOL_ADJUSTMENTS) to assert:

    0.1 µM ≤ peak ca_i ≤ 5 µM

These tests encode the calibration target from the issue #264 investigation and
will fail whenever CalciumDynamics parameters drift outside the physiological band.
"""

import pytest

from patch_sim.clamp_simulations import simulate_current_clamp
from patch_sim.constants import (
    CA1_PYRAMIDAL,
    PURKINJE,
    STN,
    THALAMIC_RELAY,
    TRN,
)
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current

# Micromolar conversion: ca_i is stored in mM; multiply by 1000 to get µM.
_MM_TO_UM = 1000.0

# Default peak ca_i band in µM (physiological single-compartment somatic
# transients).  Most presets stay below 5 µM under REPETITIVE_FIRING.
_CA_PEAK_MIN_UM_DEFAULT = 0.1
_CA_PEAK_MAX_UM_DEFAULT = 5.0

# Per-preset (min, max) overrides for the peak ca_i band.  TRN sits well
# above the default cap because the high-frequency tonic train that follows
# from the HP92-aligned conductances genuinely accumulates Ca to ~10 µM
# (cf. Cueni et al. 2008, Nat. Neurosci. 11:683 on TRN [Ca²⁺]ᵢ during LTS).
# The lower bound is tightened too: alpha_ca/tau_ca are load-bearing for
# IKCa-driven burst termination, so a downward drift that would silently
# break the burst phenotype must be caught here.
# Upper bound widened from 12 → 16 µM at correct sampling (40 kHz,
# post-#348 alignment).  Prior to the DEFAULT_SAMPLING_FREQUENCY alignment,
# the test protocol was silently stretched 2.5× longer than nominal, so
# the cell reached a quasi-steady-state during the (effective 500 ms) step
# and the peak Ca²⁺ settled around ~9 µM.  At correct (200 ms) timing the
# transient buildup phase produces a higher peak (~14.5 µM) before
# IKCa-driven AHPs and tau_ca extrusion bring it back down.  The wider
# band still covers the Cueni et al. (2008), Nat. Neurosci. 11:683
# physiological range for TRN somatic Ca during dense burst trains
# (~10–20 µM transient peaks).
_CA_PEAK_BAND_OVERRIDES: dict[str, tuple[float, float]] = {
    TRN: (5.0, 16.0),
}

# Pre/post-stimulus padding in ms common to all protocols.
_PRE_MS = 10.0
_POST_MS = 10.0

# Per-preset strong-stimulus parameters matching NEURON_PROTOCOL_ADJUSTMENTS.
_STRONG_STIM: dict[str, tuple[float, float]] = {
    PURKINJE: (10.0, 180.0),
    CA1_PYRAMIDAL: (12.0, 300.0),
    STN: (2.0, 200.0),
    TRN: (3.0, 200.0),
    THALAMIC_RELAY: (8.0, 200.0),
}


@pytest.mark.parametrize("preset_name", sorted(_STRONG_STIM))
def test_strong_stim_peak_ca_in_band(preset_name: str) -> None:
    """Peak ca_i under strong stimulation falls in the physiological band.

    Default band is [0.1, 5] µM; per-preset overrides in
    ``_CA_PEAK_BAND_OVERRIDES`` widen the cap (and tighten the floor)
    for cells whose retuned excitability legitimately drives sustained
    higher Ca that is biologically load-bearing for the preset's phenotype.

    Args:
        preset_name: Preset key from NEURON_PRESETS.
    """
    amplitude, duration = _STRONG_STIM[preset_name]
    lower_um, upper_um = _CA_PEAK_BAND_OVERRIDES.get(
        preset_name, (_CA_PEAK_MIN_UM_DEFAULT, _CA_PEAK_MAX_UM_DEFAULT)
    )
    neuron = NEURON_PRESETS[preset_name]()
    assert neuron.calcium_dynamics is not None, (
        f"Preset '{preset_name}' must have CalciumDynamics"
    )
    protocol = step_current(
        duration=_PRE_MS + duration + _POST_MS,
        current_amplitude=amplitude,
        step_start=_PRE_MS,
        step_duration=duration,
    )
    result = simulate_current_clamp(neuron, protocol)
    assert "ca_i" in result.dtype.names
    peak_um = float(result["ca_i"].max()) * _MM_TO_UM
    assert peak_um >= lower_um, (
        f"{preset_name}: peak ca_i {peak_um:.4f} µM is below {lower_um} µM"
    )
    assert peak_um <= upper_um, (
        f"{preset_name}: peak ca_i {peak_um:.4f} µM exceeds {upper_um} µM — "
        "recalibrate CalciumDynamics.alpha_ca or tau_ca for this preset"
    )
