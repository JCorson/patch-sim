"""Protocol presets and the dispatcher that builds protocol arrays.

Holds the base :data:`PROTOCOL_PRESETS` dict (clamp/timing/stimulus
parameters, keyed by protocol-preset name) and
:func:`build_protocol_from_preset`, which composes a preset with any
neuron-specific adjustments and forwards the result to the appropriate
builder.
"""

from typing import Any

import numpy as np

from patch_sim.constants import (
    ACTION_POTENTIAL,
    CURRENT_CLAMP,
    FI_CURVE,
    FREQUENCY_RESPONSE,
    HYPERPOLARIZATION_STEPS,
    IV_CURVE,
    NA_CHANNEL_ACTIVATION,
    REPETITIVE_FIRING,
    SUBTHRESHOLD_RESPONSE,
    VOLTAGE_CLAMP,
)

PROTOCOL_PRESETS: dict[str, dict[str, Any]] = {
    ACTION_POTENTIAL: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 30.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 10.0,
        "max_stimulus": 10.0,
        "stimulus_step": 0.0,
    },
    SUBTHRESHOLD_RESPONSE: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 30.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 1.5,
        "max_stimulus": 1.5,
        "stimulus_step": 0.0,
    },
    REPETITIVE_FIRING: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 180.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 15.0,
        "max_stimulus": 15.0,
        "stimulus_step": 0.0,
    },
    FI_CURVE: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 10.0,
        "stimulus_duration": 50.0,
        "post_stimulus_duration": 10.0,
        "min_stimulus": 0.0,
        "max_stimulus": 20.0,
        "stimulus_step": 2.5,
    },
    HYPERPOLARIZATION_STEPS: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 50.0,
        "stimulus_duration": 300.0,
        "post_stimulus_duration": 100.0,
        "min_stimulus": -10.0,
        "max_stimulus": -2.0,
        "stimulus_step": 2.0,
    },
    IV_CURVE: {
        "clamp_mode": VOLTAGE_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 5.0,
        "stimulus_duration": 20.0,
        "post_stimulus_duration": 5.0,
        "min_stimulus": -100.0,
        "max_stimulus": 60.0,
        "stimulus_step": 10.0,
        "holding_voltage": -70.0,
    },
    NA_CHANNEL_ACTIVATION: {
        "clamp_mode": VOLTAGE_CLAMP,
        "protocol_type": "Step",
        "pre_stimulus_duration": 5.0,
        "stimulus_duration": 20.0,
        "post_stimulus_duration": 5.0,
        "holding_voltage": -70.0,
        "min_stimulus": -60.0,
        "max_stimulus": 60.0,
        "stimulus_step": 10.0,
    },
    # Tuned so the chirp response stays subthreshold for the quiescent-at-rest
    # presets (Squid, FSI, Cortical Pyramidal, CA1) and produces a clean
    # impedance profile without any user tweaking — amp=0.25 µA/cm² gives a
    # few-mV peak-to-peak swing, small enough for the linear regime and large
    # enough for usable SNR; 1000 ms is long enough to resolve the few-Hz Ih
    # resonance band.  Autonomous pacemakers (Purkinje, SNc DA, STN, TRN) and
    # the burst-mode Thalamic Relay still need a hyperpolarizing holding
    # current; their per-neuron PROTOCOL_ADJUSTMENTS[FREQUENCY_RESPONSE]
    # overrides supply it.
    FREQUENCY_RESPONSE: {
        "clamp_mode": CURRENT_CLAMP,
        "protocol_type": "Chirp",
        "pre_stimulus_duration": 0.0,
        "stimulus_duration": 1000.0,
        "post_stimulus_duration": 0.0,
        "dc_offset": 0.0,
        "amplitude": 0.25,
        "start_frequency": 1.0,
        "end_frequency": 100.0,
    },
}

PROTOCOL_PRESET_NAMES: list[str] = list(PROTOCOL_PRESETS.keys())


def build_protocol_from_preset(
    preset_name: str,
    neuron_preset: str | None = None,
    sampling_frequency: float = 40_000.0,
    **overrides: Any,
) -> np.ndarray:
    """Build a protocol array from a named preset.

    Looks up *preset_name* in :data:`PROTOCOL_PRESETS`, applies any
    neuron-specific adjustments from :data:`NEURON_PROTOCOL_ADJUSTMENTS` when
    *neuron_preset* is supplied, then applies any caller-supplied *overrides*
    before dispatching to :func:`build_current_protocol` or
    :func:`build_voltage_protocol`.

    Args:
        preset_name: Key in :data:`PROTOCOL_PRESETS`.
        neuron_preset: Optional key in :data:`NEURON_PRESETS`.  When given,
            neuron-specific parameter adjustments are merged on top of the base
            preset before *overrides* are applied.
        sampling_frequency: Sampling frequency in Hz.  Defaults to the
            standard simulation rate (40 kHz).
        **overrides: Additional keyword arguments that override any preset or
            neuron-adjustment values.  Use builder parameter names
            (e.g. ``min_stimulus``, ``stimulus_step``).

    Returns:
        2-D array of shape ``(n_sweeps, n_samples)``.

    Raises:
        KeyError: If *preset_name* is not in :data:`PROTOCOL_PRESETS`.
        ValueError: If the resolved parameters are invalid for the chosen
            protocol type.
    """
    # Deferred imports avoid circular dependencies at module load time:
    # patch_sim.protocols.builders pulls in heavy numerical code, and
    # NEURON_PROTOCOL_ADJUSTMENTS lives in the parent package which itself
    # re-exports this function.
    from patch_sim.presets import NEURON_PROTOCOL_ADJUSTMENTS
    from patch_sim.protocols.builders import (
        build_current_protocol,
        build_voltage_protocol,
    )

    if preset_name not in PROTOCOL_PRESETS:
        raise KeyError(
            f"Unknown protocol preset {preset_name!r}. "
            f"Available: {list(PROTOCOL_PRESETS)}"
        )

    config: dict[str, Any] = dict(PROTOCOL_PRESETS[preset_name])

    if neuron_preset is not None:
        neuron_adjustments = NEURON_PROTOCOL_ADJUSTMENTS.get(neuron_preset, {})
        config.update(neuron_adjustments.get(preset_name, {}))

    config.update(overrides)

    clamp_mode: str = config.pop("clamp_mode", "Current Clamp")
    protocol_type: str = config.pop("protocol_type", "Step")

    if clamp_mode == "Current Clamp":
        return build_current_protocol(
            protocol_type=protocol_type,
            sampling_frequency=sampling_frequency,
            **config,
        )
    return build_voltage_protocol(
        protocol_type=protocol_type,
        sampling_frequency=sampling_frequency,
        **config,
    )
