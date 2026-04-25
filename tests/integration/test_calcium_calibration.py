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
from patch_sim.neuron_factory import make_neuron
from patch_sim.presets import NEURON_PRESETS
from patch_sim.protocols import step_current

# Micromolar conversion: ca_i is stored in mM; multiply by 1000 to get µM.
_MM_TO_UM = 1000.0

# Peak ca_i band in µM (physiological single-compartment somatic transients).
_CA_PEAK_MIN_UM = 0.1
_CA_PEAK_MAX_UM = 5.0

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
    """Peak ca_i under strong stimulation falls in the 0.1–5 µM physiological band.

    Args:
        preset_name: Preset key from NEURON_PRESETS.
    """
    amplitude, duration = _STRONG_STIM[preset_name]
    neuron = make_neuron(NEURON_PRESETS[preset_name])
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
    assert peak_um >= _CA_PEAK_MIN_UM, (
        f"{preset_name}: peak ca_i {peak_um:.4f} µM is below {_CA_PEAK_MIN_UM} µM"
    )
    assert peak_um <= _CA_PEAK_MAX_UM, (
        f"{preset_name}: peak ca_i {peak_um:.4f} µM exceeds {_CA_PEAK_MAX_UM} µM — "
        "recalibrate CalciumDynamics.alpha_ca or tau_ca for this preset"
    )
